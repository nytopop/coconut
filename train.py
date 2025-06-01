import argparse
import io
import itertools
import logging
import math
import os
from datetime import datetime

import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from evotorch import Problem
from evotorch.algorithms import CEM, CMAES, SNES, XNES
from evotorch.decorators import vectorized
from evotorch.logging import PandasLogger
from kokoro import KModel
from misaki import en, espeak
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import highpass_biquad, lowpass_biquad
from transformers import MimiModel

from audio_embed import VoiceEncoder, preprocess_wav
from loss import mimi_loss, resemblyzer_loss, spectral_loss

logging.basicConfig(level=logging.INFO)


def main():
    desc = "Evolving voices for Kokoro with various optimization algorithms."
    d = " [%(default)s]"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--out-dir", default="out", metavar="DIR", help="output directory" + d)
    parser.add_argument("--suffix", help="suffix to append to output path" + d)
    parser.add_argument("--save-every", type=int, default=25, metavar="N", help="save every N iterations" + d)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed" + d)
    parser.add_argument("-d", default="cuda", choices=["cpu", "cuda"], help=d)

    ha = parser.add_argument_group("optimization algorithm")
    ha.add_argument("--alg", default="cmaes", choices=["xnes", "snes", "cmaes", "cem"], help=d)
    ha.add_argument("--sigma", type=float, default=0.01, metavar="σ", help="initial step size" + d)
    ha.add_argument("--pop", type=int, metavar="λ", help="population size [auto]")
    ha.add_argument("--rho", type=float, default=0.1, metavar="ρ", help="parenthood ratio (CEM only)" + d)
    ha.add_argument("--spectral", default=False, action="store_true", help="add spectral loss term" + d)

    # TODO: config the optimizer of xnes & snes (we can use momentum based adam/clipup)

    hb = parser.add_argument_group("batching")
    hb.add_argument("--chunk", type=int, default=24, metavar="N", help="concurrency (min = batch)" + d)
    hb.add_argument("--batch", type=int, default=4, metavar="N", help="regularization minibatch size" + d)
    hb.add_argument("--k", type=int, default=1, metavar="N", help="iterations per minibatch" + d)

    hd = parser.add_argument_group("dataset")
    hd.add_argument("--dataset", choices=["expresso", "expresso-conv", "animevox", "genshin"], required=True)
    hd.add_argument("--speaker", required=True, help=d)
    hd.add_argument("--style", help=d)
    hd.add_argument("--no-stream", default=False, action="store_true", help="download dataset" + d)

    ho = parser.add_argument_group("objectives")
    ho.add_argument("--interp", metavar="PATH", help="fit a linear combination of any voices at PATH")
    ho.add_argument("--bias", default=False, action="store_true", help="fit a free bias term" + d)
    args = parser.parse_args()

    if args.interp is None and not args.bias:
        parser.error("at least one objective is required: --interp, --bias")

    # mimi for calculating perceptual loss
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(args.d)

    # resemblyzer speaker encoding model
    venc = VoiceEncoder(device=args.d)

    # configure & load kokoro
    try:
        fallback = espeak.EspeakFallback(british=False)
    except Exception:
        fallback = None

    g2p = en.G2P(trf=False, british=False, fallback=fallback, unk="")
    tts = KModel(repo_id="hexgrad/Kokoro-82M").to(device=args.d)

    # configure & load dataset
    streaming = not args.no_stream

    def row_filter(r) -> bool:
        return r["transcription"] != "" and r["speaker"] == args.speaker

    match args.dataset:
        case "expresso":
            ds = load_dataset("ylacombe/expresso", split="train", streaming=streaming)
            ds = ds.rename_columns({"text": "transcription", "speaker_id": "speaker"})
            if args.style is not None:
                ds = ds.filter(lambda r: row_filter(r) and r["style"] == args.style)
            else:
                ds = ds.filter(row_filter)
        case "expresso-conv":
            ds = load_dataset("nytopop/expresso-conversational", split="train", streaming=streaming)
            ds = ds.rename_columns({"text": "transcription", "speaker_id": "speaker"})
            if args.style is not None:
                ds = ds.filter(lambda r: row_filter(r) and r["style"] == args.style)
            else:
                ds = ds.filter(row_filter)
        case _ if args.style:
            parser.error("--style is only valid for expresso and expresso-conv datasets")
        case "animevox":
            ds = load_dataset("taresh18/AnimeVox", split="train", streaming=streaming)
            ds = ds.rename_columns({"character_name": "speaker"})
            ds = ds.filter(row_filter)
        case "genshin":
            ds = load_dataset("simon3000/genshin-voice", split="train", streaming=streaming)
            ds = ds.filter(
                lambda r: row_filter(r)
                and "{" not in r["transcription"]
                and r["language"] == "English(US)"
                and r["type"] == "Dialog"
            )

    if not streaming:
        ds = ds.shuffle(seed=args.seed)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=4,
        collate_fn=lambda rows: {
            "audio": [row["audio"] for row in rows],
            "phonemes": [g2p(row["transcription"].replace("*", ""))[0][:508] for row in rows],
        },
    )

    # configure objective function
    if args.interp:
        try:
            pack = [torch.load(f"{args.interp}/{p}", map_location=args.d) for p in os.listdir(args.interp)]
        except NotADirectoryError:
            pack = [torch.load(f"{args.interp}", map_location=args.d)]
        n, pack = len(pack), torch.stack(pack)  # => [N, 510, 1, 256]
    else:
        n = 0

    init = torch.zeros(n + 256 if args.bias else n)

    def i2v(pop: torch.Tensor) -> torch.Tensor:
        scaled = biased = 0

        if args.interp:
            weight = torch.softmax(pop[:, :n][..., None, None, None], dim=1)
            scaled = (pack * weight).sum(dim=1)

        # TODO: interpolate the final bias from a stack of equidistant buckets
        if args.bias:
            biased = pop[:, n:][:, None, None, :].expand(-1, 510, -1, -1)

        return scaled + biased

    @vectorized
    def loss(pop: torch.Tensor) -> torch.Tensor:
        pop_loss = []
        styles = i2v(pop.to(tts.device))  # => [P, 510, 1, 256]
        N = max(1, args.chunk // len(phonemes))  # max candidates per forward pass

        for styles in [styles[i : i + N] for i in range(0, styles.shape[0], N)]:
            n = styles.shape[0]

            styles = styles.repeat_interleave(len(phonemes), 0)  # => [C, 510, 1, 256]

            with torch.autocast(tts.device.type):
                syn_wavs, syn_lens = tts.forward_batch(styles, phonemes * n, speed=1)

            pool, temp = mimi_loss(mimi, ref_wavs, ref_lens, syn_wavs, syn_lens, n)

            resm = 1e-2 * resemblyzer_loss(venc, ref_wavs, ref_lens, syn_wavs, syn_lens, n)

            loss = pool + temp + resm

            if args.interp:
                # TODO: add a regularization term of distance to nearest in pack
                pass

            if args.spectral:
                loss += spectral_loss(ref_wavs, syn_wavs, n)

            pop_loss.append(1e2 * loss)

        pop_loss = torch.cat(pop_loss)

        return torch.nan_to_num(pop_loss, nan=math.inf)

    # configure optimizer & search algorithm (TODO: could do device=args.d, but do we need to? uses vram)
    p = Problem(
        "min", loss, solution_length=init.shape[0], initial_bounds=(-args.sigma, args.sigma), seed=args.seed
    )

    match args.alg:
        case "xnes":
            s = XNES(p, stdev_init=args.sigma, popsize=args.pop, center_init=init)
        case "snes":
            s = SNES(p, stdev_init=args.sigma, popsize=args.pop, center_init=init)
        case "cmaes":
            s = CMAES(p, stdev_init=args.sigma, popsize=args.pop, center_init=init)
        case "cem":
            if args.pop is None:
                args.pop = 4 + 3 * math.log(n)
            s = CEM(p, stdev_init=args.sigma, popsize=args.pop, parenthood_ratio=args.rho, center_init=init)

    pop = getattr(s, "_popsize", None)
    if pop is None:
        pop = getattr(s, "popsize", None)

    objective = "joint" if args.interp and args.bias else "interp" if args.interp else "bias"

    title = (
        f"dataset={args.dataset} speaker={args.speaker}"
        + (f" style={args.style}" if args.style else "")
        + f"\nobj={objective} alg={args.alg} σ={args.sigma} λ={pop} k={args.k} batch={args.batch}"
    )

    log = PandasLogger(s)

    def process_audio(row, to_sr: int = 16000) -> np.ndarray:
        if "array" in row:
            au, sr = row["array"], row["sampling_rate"]
        else:
            au, sr = lr.load(io.BytesIO(row["bytes"]))
        return preprocess_wav(au, source_sr=sr, to_sr=to_sr)

    dt_started = datetime.now()
    date_string = dt_started.strftime("%Y-%m-%d-%H:%M")
    exp_dir = f"{args.out_dir}/{date_string}" + (f"-{args.suffix}" if args.suffix is not None else "")

    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"experiment dir: {exp_dir}")

    def save_checkpoint(i):
        best = s.status["pop_best"].access_values(keep_evals=True).unsqueeze(0).to(tts.device)
        best = i2v(best).squeeze(0)

        # for best quality, generate using non-batched forward()
        for j, span in enumerate(phonemes):
            out = tts.forward(span, best[len(span) - 1], speed=1)
            sf.write(f"{exp_dir}/checkpoint-{i}-{j}.wav", out, 24000)

        torch.save(best, f"{exp_dir}/checkpoint-{i}.pt")

    iters = itertools.count(1)

    for epoch in itertools.count(0):
        for batch in dataloader:
            ref_wavs_ = [
                torch.from_numpy(process_audio(row, to_sr=24000)).float().to(args.d) for row in batch["audio"]
            ]
            ref_wavs = pad_sequence(ref_wavs_, batch_first=True)
            ref_wavs = highpass_biquad(ref_wavs, 24000, 90)
            ref_lens = torch.tensor([ref.shape[0] for ref in ref_wavs_]).to(args.d)

            phonemes = batch["phonemes"]

            # perform K steps over the same minibatch before swapping to a new one
            for i in itertools.islice(iters, args.k):
                s.step()

                if i == 1 or i % args.save_every == 0:
                    logging.info(
                        f"ep={epoch} it={i:04} ∧={s.status['best_eval']:.3e} λ∧={s.status['pop_best_eval']:.3e} μ={s.status['mean_eval']:.4e} t+{str(datetime.now() - dt_started)}"
                    )
                    save_checkpoint(i)
                    log.to_dataframe().plot(title=title)
                    plt.savefig(f"{exp_dir}/train.png")
                    plt.close()


if __name__ == "__main__":
    main()
