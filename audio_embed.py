import functools
import struct
from pathlib import Path
from typing import List, Optional, Union

import librosa as lr
import numpy as np
import torch
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
from torch import nn
from torchaudio.transforms import MelSpectrogram

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10  # In milliseconds
mel_n_channels = 40

## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160  # 1600 ms

## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds

# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out.
vad_moving_average_width = 8

# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6

## Audio volume normalization
audio_norm_target_dBFS = -30

## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3


class VoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device] = None, weights_fpath: Union[Path, str] = None):
        """
        If None, defaults to cuda if it is available on your machine, otherwise the model will
        run on cpu. Outputs are always returned on the cpu, as numpy arrays.
        :param weights_fpath: path to "<CUSTOM_MODEL>.pt" file path.
        If None, defaults to built-in "pretrained.pt" model
        """
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model'speaker weights
        if weights_fpath is None:
            weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        else:
            weights_fpath = Path(weights_fpath)

        if not weights_fpath.exists():
            raise Exception("Couldn't find the voice encoder pretrained model at %s." % weights_fpath)
        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        msg = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=int(sampling_rate * mel_window_length / 1000),
            hop_length=int(sampling_rate * mel_window_step / 1000),
            n_mels=mel_n_channels,
        )
        self.mel_spectrogram = msg.to(device)

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size).
        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
        """
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    @functools.cache
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to
        obtain partial utterances of <partials_n_frames> each. Both the waveform and the
        mel spectrogram slices are returned, so as to make each partial utterance waveform
        correspond to its spectrogram.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wav_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 < min_coverage <= 1

        # Compute how many frames separate two partial utterances
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % (
            sampling_rate / (samples_per_frame * partials_n_frames)
        )

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    @torch.no_grad()
    def embed(self, wavs: List[torch.FloatTensor], rate=1.3, min_coverage=0.75) -> torch.FloatTensor:
        """
        Computes an embedding for a batch of utterances. Each utterance is divided to partial slices
        and an embedding is computed for each. The complete embedding for each utterance is the l2 norm
        of the average embedding of all partials.

        :param wavs: preprocessed utterance waveforms as a list of torch tensors
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the embedding as a B x N torch tensor where N is the model_embedding_size.
        """
        mels, nmel = [], []

        # compute mel spectrograms of partial slices of the input wavs
        for wav in wavs:
            wav = wav.to(self.device)

            _wav, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
            nmel.append(len(mel_slices))

            if _wav[-1].stop >= len(wav):
                wav = torch.nn.functional.pad(wav, (0, _wav[-1].stop - len(wav)))

            mel = self.mel_spectrogram(wav).transpose(1, 0)
            mels.extend([mel[s] for s in mel_slices])

        # run mels through the pretrained model to get partial embeddings of each mel slice
        partial_embeds = self(torch.stack(mels))

        # compute the mean of partial embeddings belonging to each input wav
        lens = torch.cumsum(torch.tensor([0] + nmel), dim=0)
        sums = torch.stack([torch.sum(partial_embeds[a:b], axis=0) for a, b in zip(lens, lens[1:])])
        mean = sums / torch.tensor(nmel, dtype=torch.float, device=self.device).unsqueeze(1)

        # L2 norm
        norm = mean / torch.linalg.vector_norm(mean, dim=1).unsqueeze(1)

        return norm


def preprocess_wav(
    fpath_or_wav: Union[str, Path, np.ndarray], to_sr: int = sampling_rate, source_sr: Optional[int] = None
):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that
    The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform'speaker sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = lr.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav
    if source_sr is not None:
        wav = lr.resample(wav, orig_sr=source_sr, target_sr=to_sr)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)

    return wav


int16_max = (2**15) - 1


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(
            vad.is_speech(pcm_wave[window_start * 2 : window_end * 2], sample_rate=sampling_rate)
        )
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1 :] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]
