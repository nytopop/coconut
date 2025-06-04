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
from evotorch.algorithms import CEM, CMAES, SNES, XNES, Cosyne
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

    ha = parser.add_argument_group("optimizer")
    ha.add_argument("--alg", default="cmaes", choices=["xnes", "snes", "cmaes", "cem", "cosyne"], help=d)
    ha.add_argument("--sigma", type=float, default=0.01, metavar="σ", help="initial step size" + d)
    ha.add_argument("--rho", type=float, default=0.1, metavar="ρ", help="parenthood ratio (CEM only)" + d)
    ha.add_argument("--pop", type=int, default=24, metavar="λ", help="population size (candidates/iter)" + d)
    ha.add_argument("--window", type=int, default=4, metavar="W", help="window size (examples/iter)" + d)
    ha.add_argument("--stride", type=int, metavar="S", help="sliding window stride [W/2]")
    ha.add_argument("--batch", type=int, default=24, metavar="B", help="batch size" + d)
    ha.add_argument("--spectral", default=False, action="store_true", help="add spectral loss term" + d)

    hd = parser.add_argument_group("dataset")
    hd.add_argument("--dataset", choices=["expresso", "expresso-conv", "animevox", "genshin"], required=True)
    hd.add_argument("--speaker", required=True, help=d)
    hd.add_argument("--style", help=d)
    hd.add_argument("--no-stream", default=False, action="store_true", help="download dataset" + d)
    hd.add_argument("--epochs", type=int, help="epoch limit" + d)

    ho = parser.add_argument_group("objectives")
    ho.add_argument("--interp", metavar="PATH", help="fit a linear combination of voice(s) at PATH")
    ho.add_argument("--bias", default=False, action="store_true", help="fit a free bias term" + d)

    args = parser.parse_args()

    if args.interp is None and not args.bias:
        parser.error("at least one objective is required: --interp, --bias")

    if args.stride is not None and args.stride > args.window:
        parser.error(f"stride {args.stride} > window {args.window} size; this is probably not what you want")

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
    tts = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to(device=args.d)

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

    stride = max(1, int(args.window // 2) if args.stride is None else args.stride)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=stride,
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
        n, pack = len(pack), torch.stack(pack)  # => [V, 510, 1, 256]
    else:
        n = 0

    init = torch.zeros(n + 256 if args.bias else n)

    def i2v(pop: torch.Tensor) -> torch.Tensor:
        scaled = biased = bias_q = 0

        if args.interp:
            weight = torch.softmax(pop[:, :n][..., None, None, None], dim=1)
            scaled = (pack * weight).sum(dim=1)

        # TODO: interpolate the final bias from a stack of equidistant buckets
        if args.bias:
            bias_q = pop[:, n:].pow(2).mean(dim=1)
            biased = pop[:, n:][:, None, None, :].expand(-1, 510, -1, -1)

        return scaled + biased, bias_q

    i = 1

    @vectorized
    def loss(pop: torch.Tensor) -> torch.Tensor:
        pop_loss = []
        styles, bias_q = i2v(pop.to(tts.device))  # => [P, 510, 1, 256], [P]
        N = max(1, args.batch // len(phonemes))  # max candidates per forward pass

        for styles in [styles[i : i + N] for i in range(0, styles.shape[0], N)]:
            n, _, _, _ = styles.shape

            styles = styles.repeat_interleave(len(phonemes), 0)  # => [n, 510, 1, 256]

            with torch.autocast(tts.device.type):
                syn_wavs, syn_lens = tts.forward_batch(styles, phonemes * n, speed=1)

            pool, temp = mimi_loss(mimi, ref_wavs, ref_lens, syn_wavs, syn_lens, n)

            loss = pool + temp

            loss += 1e-2 * resemblyzer_loss(venc, ref_wavs, ref_lens, syn_wavs, syn_lens, n)

            if args.spectral:
                loss += 1e-5 * spectral_loss(ref_wavs, syn_wavs, n)

            pop_loss.append(loss)

        pop_loss = torch.cat(pop_loss)

        if args.interp and args.bias and pack.shape[0] > 1:
            pop_loss += 3e-5 * bias_q

        return 1e2 * torch.nan_to_num(pop_loss, nan=math.inf)

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
        case "cosyne":
            if args.pop is None:
                args.pop = 4 + 3 * math.log(n)
            s = Cosyne(
                p, popsize=args.pop, tournament_size=4, mutation_stdev=args.sigma, elitism_ratio=args.rho
            )

    pop = getattr(s, "_popsize", None)
    if pop is None:
        pop = getattr(s, "popsize", None)

    objective = "joint" if args.interp and args.bias else "interp" if args.interp else "bias"

    title = (
        f"dataset={args.dataset} speaker={args.speaker}"
        + (f" style={args.style}" if args.style else "")
        + f"\nobj={objective} alg={args.alg} σ={args.sigma} λ={pop} w={args.window} s={stride}"
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
        best = i2v(best)[0].squeeze(0)

        # for best quality, generate using non-batched forward()
        for j, span in enumerate(phonemes):
            out = tts.forward(span, best[len(span) - 1], speed=1)
            sf.write(f"{exp_dir}/checkpoint-{i}-{j}.wav", out, 24000)

        torch.save(best, f"{exp_dir}/checkpoint-{i}.pt")

    b_aud = []
    phonemes = []

    for epoch in itertools.count(0):
        for batch in dataloader:
            for au, ph in zip([process_audio(au, to_sr=24000) for au in batch["audio"]], batch["phonemes"]):
                if len(au) > 0:
                    b_aud.append(au)
                    phonemes.append(ph)

            b_aud = b_aud[-args.window :]
            phonemes = phonemes[-args.window :]

            ref_wavs_ = [torch.from_numpy(au).float().to(args.d) for au in b_aud]
            ref_wavs = pad_sequence(ref_wavs_, batch_first=True)
            ref_wavs = highpass_biquad(ref_wavs, 24000, 90)
            ref_lens = torch.tensor([ref.shape[0] for ref in ref_wavs_]).to(args.d)

            s.step()

            if i == 1 or i % args.save_every == 0:
                logging.info(
                    f"ep={epoch} it={i:04} ∧={s.status['best_eval']:.3e} λ∧={s.status['pop_best_eval']:.3e} μ={s.status['mean_eval']:.4e} t+{str(datetime.now() - dt_started)}"
                )
                save_checkpoint(i)
                log.to_dataframe().plot(title=title)
                plt.savefig(f"{exp_dir}/train.png")
                plt.close()

            i += 1

        if args.epochs is not None:
            if (epoch + 1) >= args.epochs:
                if (i - 1) % args.save_every != 0:
                    logging.info(
                        f"ep={epoch} it={i:04} ∧={s.status['best_eval']:.3e} λ∧={s.status['pop_best_eval']:.3e} μ={s.status['mean_eval']:.4e} t+{str(datetime.now() - dt_started)}"
                    )
                    save_checkpoint(i)
                    log.to_dataframe().plot(title=title)
                    plt.savefig(f"{exp_dir}/train.png")
                    plt.close()
                break


if __name__ == "__main__":
    main()
