from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import MimiModel


def joint_mimi_spectral_loss(
    mimi: MimiModel,
    ref_wavs_: List[torch.Tensor],  # [B, ~T]
    syn_wavs: torch.Tensor,  # [B x N, T]
    syn_lens: torch.Tensor,  # [B x N]
    n: int,
) -> torch.Tensor:  # [N]
    ref_wavs = pad_sequence(ref_wavs_, batch_first=True)
    ref_lens = torch.tensor([ref.shape[0] for ref in ref_wavs_]).to(ref_wavs.device)

    pool, temp = mimi_loss(mimi, ref_wavs, ref_lens, syn_wavs, syn_lens, n)

    spec = 1e-5 * spectral_loss(ref_wavs, syn_wavs, n)

    return pool + temp + spec


def mimi_loss(
    mimi: MimiModel,
    ref_wavs: torch.Tensor,  # [B, Tr]
    ref_lens: torch.Tensor,  # [B]
    syn_wavs: torch.Tensor,  # [B x N, Ts]
    syn_lens: torch.Tensor,  # [B x N]
    n: int,
) -> torch.Tensor:  # [N]
    # TODO: could cache ref_* but it's fast enough alr
    # [B, Qr]
    ref_pool, ref_temp, ref_lens = mimi_embed(mimi, ref_wavs, ref_lens)

    # [B x N, Qs]
    syn_pool, syn_temp, syn_lens = mimi_embed(mimi, syn_wavs, syn_lens)

    # 1. truncate ref_temp/syn_temp and ref_lens/syn_lens to the minimum of Qr and Qs
    #    (overall padded dim may be different)
    Q_rs = min(ref_temp.shape[1], syn_temp.shape[1])
    ref_temp = ref_temp[:, :Q_rs, :]
    syn_temp = syn_temp[:, :Q_rs, :]

    # 2. repeat ref_* batch dimension by N to align with syn
    ref_pool = ref_pool.repeat(n, 1)
    ref_temp = ref_temp.repeat(n, 1, 1)
    ref_lens = ref_lens.repeat(n)

    # 3. calculate mean squared error between each Qrs embedding, producing [B x N, Qrs]
    mse_per_timestep = (ref_temp - syn_temp).pow(2).mean(dim=-1)

    # 4. calculate mean mean mse up to min(ref_lens, syn_lens) at each position, producing [B x N]
    #    (avoid comparing embeds of padded positions)
    lens = torch.minimum(ref_lens, syn_lens)  # [B x N]
    indices = torch.arange(Q_rs, device=mse_per_timestep.device)
    mask = indices.unsqueeze(0) < lens.unsqueeze(1)  # [B x N, Qrs]
    temp_mse_per_sample = (mse_per_timestep * mask).sum(dim=1) / lens  # [B x N]
    temp_mse = 1e-2 * temp_mse_per_sample.reshape(n, -1).mean(dim=1)  # [N]

    # 5. calculate mean mse of pooled embeddings, producing [N]
    pool_mse_per_sample = (ref_pool - syn_pool).pow(2).mean(dim=1)  # [B x N]
    pool_mse = pool_mse_per_sample.reshape(n, -1).mean(dim=1)  # [N]

    return pool_mse, temp_mse


def mimi_embed(
    mimi: MimiModel,
    wavs: torch.Tensor,  # [B, T]
    lens: torch.Tensor,  # [B]
) -> Tuple[
    torch.Tensor,  # [B, 512] (pooled embeddings)
    torch.Tensor,  # [B, Q, 512] where T/Q=960 (temporal embeddings, use with lens)
    torch.Tensor,  # [B] (lens)
]:
    with torch.inference_mode():
        B, T = wavs.shape
        encode = mimi.encoder(wavs.unsqueeze(1)).transpose(1, 2)  # [B, Q, 512] where T/Q=960

        lens = torch.ceil(lens / 960).long()  # [B]
        _, Q, _ = encode.shape
        mask = torch.arange(Q, device=encode.device).unsqueeze(0) < lens.unsqueeze(1)  # [B, Q]

        pool = (encode * mask.unsqueeze(-1)).sum(dim=1) / lens.unsqueeze(-1)
        pool = F.normalize(pool, p=2, dim=1)

        temp = mimi.encoder_transformer(encode, attention_mask=mask)[0]  # [B, Q, 512]
        temp = F.normalize(temp, p=2, dim=2)

    return pool, temp, lens


# ware; here be vibe coded
def spectral_loss(
    ref_audio: torch.Tensor,  # [B_ref, T_ref_padded]
    syn_audio: torch.Tensor,  # [B_syn (B_ref * n), T_syn]
    n_candidate_variations: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:  # [n]
    """
    Calculates a simple L1 loss on log-magnitude spectrograms.
    ref_audio: Padded reference waveforms.
    syn_audio: Synthesized waveforms.
    n_candidate_variations: Number of synthetic variations per reference.
    """
    device = ref_audio.device

    # 1. Ensure consistent audio lengths for STFT
    # T_ref_padded can be different from T_syn
    T_common = min(ref_audio.shape[1], syn_audio.shape[1])
    ref_audio_trimmed = ref_audio[:, :T_common]
    syn_audio_trimmed = syn_audio[:, :T_common]

    # 2. Repeat reference audio to match the synthetic batch dimension structure
    # ref_audio_trimmed is [B_ref, T_common]
    # We want [B_ref * n, T_common] to match syn_audio_trimmed
    ref_audio_repeated = ref_audio_trimmed.repeat_interleave(n_candidate_variations, dim=0)
    # Now ref_audio_repeated is [B_ref * n, T_common]
    # And syn_audio_trimmed is [B_ref * n, T_common]

    # 3. Calculate STFT
    window = torch.hann_window(win_length, device=device)

    ref_stft = torch.stft(
        ref_audio_repeated,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        normalized=False,  # Using False to be closer to common STFT implementations
    )
    syn_stft = torch.stft(
        syn_audio_trimmed,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        normalized=False,
    )

    # Ensure same number of frames (can sometimes be off by one due to padding/length)
    min_frames = min(ref_stft.shape[-1], syn_stft.shape[-1])
    ref_stft = ref_stft[..., :min_frames]
    syn_stft = syn_stft[..., :min_frames]

    # 4. Calculate log magnitudes
    # Adding a small epsilon for numerical stability before log
    ref_log_mag = torch.log(torch.abs(ref_stft) + 1e-7)
    syn_log_mag = torch.log(torch.abs(syn_stft) + 1e-7)

    # 5. Calculate L1 loss between log magnitudes
    # This will be [B_ref * n, Freq_bins, Time_frames]
    loss_per_frame = F.l1_loss(ref_log_mag, syn_log_mag, reduction="none")

    # 6. Average loss over frequency and time dimensions to get per-sample loss
    # Resulting shape: [B_ref * n]
    loss_per_sample = loss_per_frame.mean(dim=[1, 2])  # Mean over Freq_bins and Time_frames

    # 7. Reshape and average to get the final loss of shape [n]
    # (Average over the B_ref dimension for each of the n variations)
    # loss_per_sample is [B_ref * n_candidate_variations]
    # We want to group by n_candidate_variations and average over B_ref
    final_loss = loss_per_sample.view(n_candidate_variations, -1).mean(dim=1)

    return final_loss
