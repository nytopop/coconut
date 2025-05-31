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
from evotorch.algorithms import CMAES, SNES, XNES, CEM
from evotorch.decorators import vectorized
from evotorch.logging import PandasLogger
from kokoro import KModel
from misaki import en, espeak
from torchaudio.functional import resample

from audio_embed import VoiceEncoder, preprocess_wav

logging.basicConfig(level=logging.INFO)


def main():
    desc = "Evolving voices for Kokoro with various optimization algorithms."
    d = " [%(default)s]"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--out-dir", default="out", metavar="DIR", help="output directory" + d)
    parser.add_argument("--suffix", help="suffix to append to output path" + d)
    parser.add_argument("--save-every", type=int, default=20, metavar="N", help="save every N iterations" + d)
    # TODO: save optimizer state so we can do warm resume (with stuff like covariance matrix prepared)
    # parser.add_argument("--resume", metavar="PATH", help="resume from a saved checkpoint" + d)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed" + d)
    parser.add_argument("-t", default="fp16", choices=["fp32", "fp16", "bf16"], help=d)
    parser.add_argument("-d", default="cuda", choices=["cpu", "cuda"], help=d)

    ha = parser.add_argument_group("optimization algorithm")
    ha.add_argument("--alg", default="snes", choices=["xnes", "snes", "cmaes", "cem"], help=d)
    ha.add_argument("--sigma", type=float, default=0.01, metavar="σ", help="initial step size" + d)
    ha.add_argument("--pop", type=int, metavar="λ", help="population size [auto]")
    ha.add_argument("--rho", type=float, default=0.1, metavar="ρ", help="parenthood ratio (CEM only)" + d)
    # TODO: config the optimizer of xnes & snes (we can use momentum based adam/clipup)

    hb = parser.add_argument_group("batching")
    hb.add_argument("--chunk", type=int, default=24, metavar="N", help="concurrency (min = batch)" + d)
    hb.add_argument("--batch", type=int, default=8, metavar="N", help="regularization minibatch size" + d)
    hb.add_argument("--k", type=int, default=4, metavar="N", help="iterations per minibatch" + d)

    hd = parser.add_argument_group("dataset")
    hd.add_argument("--dataset", choices=["expresso", "animevox", "genshin"], required=True)
    hd.add_argument("--speaker", required=True, help=d)
    hd.add_argument("--style", help=d)
    hd.add_argument("--no-stream", default=False, action="store_true", help="download dataset" + d)

    ho = parser.add_argument_group("objectives (choose one)")
    ho.add_argument("--blend", metavar="DIR", help="linear combination of all voices in DIR")
    ho.add_argument("--bias", metavar="PATH", help="style + ∧(bias) for style in voicepack")
    ho.add_argument("--bias-mean", metavar="PATH", help="μ(voicepack) + ∧(bias) ")

    args = parser.parse_args()

    venc = VoiceEncoder(device=args.d)

    if [args.blend, args.bias, args.bias_mean].count(None) != 2:
        parser.error("exactly one objective is required: --blend, --bias, --bias-mean")

    # configure & load kokoro
    try:
        fallback = espeak.EspeakFallback(british=False)
    except Exception:
        fallback = None

    g2p = en.G2P(trf=True, british=False, fallback=fallback, unk="")

    match args.t:
        case "fp16":
            ty, cx = torch.float16, True
        case "bf16":
            ty, cx = torch.bfloat16, False
        case "fp32":
            ty, cx = torch.float32, False

    tts = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=cx).to(dtype=ty, device=args.d)

    # configure & load dataset
    streaming = not args.no_stream

    def row_filter(r) -> bool:
        return r["transcription"] != "" and r["speaker"] == args.speaker

    match args.dataset:
        case "expresso":
            ds = load_dataset("nytopop/expresso-conversational", split="train", streaming=streaming)
            ds = ds.rename_columns({"text": "transcription", "speaker_id": "speaker"})
            if args.style is not None:
                ds = ds.filter(lambda r: row_filter(r) and r["style"] == args.style)
            else:
                ds = ds.filter(row_filter)
        case _ if args.style:
            parser.error("--style is only valid for the expresso dataset")
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
            "phonemes": [g2p(row["transcription"])[0] for row in rows],
        },
    )

    # configure objective function
    objective = "blend" if args.blend else "bias" if args.bias else "bias-mean"

    match objective:
        case "blend":
            vpacks = [torch.load(f"{args.blend}/{p}", map_location=args.d) for p in os.listdir(args.blend)]
            vpacks = torch.stack(vpacks)  # => [N, 510, 1, 256]
            n = vpacks.shape[0]
            bounds = (-1 / n, 1 / n)

            def i2v(pop: torch.Tensor) -> torch.Tensor:
                weights = torch.softmax(pop[..., None, None, None], dim=1)
                return (vpacks * weights).sum(dim=1)
        case "bias":
            vpack = torch.load(args.bias, map_location=args.d)  # => [510, 1, 256]
            n, bounds = 256, (-args.sigma, args.sigma)

            def i2v(pop: torch.Tensor) -> torch.Tensor:
                bias = pop[:, None, None, :].expand(-1, 510, -1, -1)
                return bias + vpack
        case "bias-mean":
            vpack = torch.load(args.bias_mean, map_location=args.d)  # => [510, 1, 256]
            voice = torch.mean(vpack, dim=0)  # => [1, 256]
            n, bounds = 256, (-args.sigma, args.sigma)

            def i2v(pop: torch.Tensor) -> torch.Tensor:
                biased = (pop.unsqueeze(1) + voice).unsqueeze(1)
                return biased.expand(-1, 510, -1, -1)

    @vectorized
    def loss(pop: torch.Tensor) -> torch.Tensor:
        pop_loss = []
        styles = i2v(pop.to(tts.device))  # => [P, 510, 1, 256]
        N = max(1, args.chunk // len(phonemes))  # max candidates per forward pass

        for styles in [styles[i : i + N] for i in range(0, styles.shape[0], N)]:
            n = styles.shape[0]

            # prepare batch data for tts
            styles = styles.repeat_interleave(len(phonemes), 0)  # => [C, 510, 1, 256]

            # generate audio + resample + trim lengths
            outs, durs = tts.forward_batch(styles, phonemes * n, speed=1.0)
            outs = resample(outs, orig_freq=24000, new_freq=16000)
            trimmed = [outs[i, : int(durs[i] / 1.5)] for i in range(0, outs.shape[0])]

            # compute embeddings
            all_embeds = venc.embed(trimmed).reshape(n, len(phonemes), 256)  # => [N, B, 256]
            M = torch.tensor([len(phonemes)] * n, device=venc.device).unsqueeze(1)
            mean = torch.sum(all_embeds, dim=1) / M
            embs = mean / torch.linalg.vector_norm(mean, dim=1).unsqueeze(1)

            # and loss is just sum of squared error
            loss = torch.sum(torch.square(embs - target_embed.unsqueeze(0)), dim=1)
            pop_loss.append(loss)

        pop_loss = torch.cat(pop_loss)

        return torch.nan_to_num(pop_loss, nan=math.inf)

    # configure optimizer & search algorithm (TODO: could do device=args.d, but do we need to? uses vram)
    p = Problem("min", loss, solution_length=n, initial_bounds=bounds, seed=args.seed)

    match args.alg:
        case "xnes":
            s = XNES(p, stdev_init=args.sigma, popsize=args.pop)
        case "snes":
            s = SNES(p, stdev_init=args.sigma, popsize=args.pop)
        case "cmaes":
            s = CMAES(p, stdev_init=args.sigma, popsize=args.pop)
        case "cem":
            if args.pop is None:
                args.pop = 4 + 3 * math.log(n)
            s = CEM(p, stdev_init=args.sigma, popsize=args.pop, parenthood_ratio=args.rho)

    pop = getattr(s, "_popsize", None)
    if pop is None:
        pop = getattr(s, "popsize", None)

    title = (
        f"dataset={args.dataset} speaker={args.speaker}"
        + (f" style={args.style}" if args.style else "")
        + f"\n{objective} alg={args.alg} σ={args.sigma} λ={pop} k={args.k} batch={args.batch}"
    )

    log = PandasLogger(s)

    def process_audio(row) -> np.ndarray:
        if "array" in row:
            au, sr = row["array"], row["sampling_rate"]
        else:
            au, sr = lr.load(io.BytesIO(row["bytes"]))
        return preprocess_wav(au, source_sr=sr)

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    exp_dir = f"{args.out_dir}/{date_string}" + (f"-{args.suffix}" if args.suffix is not None else "")

    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Experiment dir: {exp_dir}")

    # TODO: shouldn't we be using the population best, not the best?
    def save_checkpoint(i):
        best = s.status["pop_best"].access_values(keep_evals=True).unsqueeze(0).to(tts.device)
        best = i2v(best).squeeze(0)

        # for best quality, generate using non-batched forward()
        for j, span in enumerate(phonemes):
            out = tts.forward(span, best[len(span) - 1], speed=1.0)
            sf.write(f"{exp_dir}/checkpoint-{i}-{j}.wav", out, 24000)

        torch.save(best, f"{exp_dir}/checkpoint-{i}.pt")

    iters = itertools.count(1)

    for epoch in itertools.count(0):
        for batch in dataloader:
            # compute the L2 normed mean embedding of everything in batch["audio"]
            audio = [torch.from_numpy(process_audio(row)).float().to(args.d) for row in batch["audio"]]
            target_embed_all = venc.embed(audio)
            mean = torch.sum(target_embed_all, dim=0) / target_embed_all.shape[0]

            # NOTE: these are accessed by loss() (in s.step()) and save_checkpoint() (kinda jank but it works)
            target_embed = mean / torch.linalg.vector_norm(mean)
            phonemes = batch["phonemes"]

            # perform K steps over the same minibatch before swapping to a new one
            for i in itertools.islice(iters, args.k):
                s.step()

                if i % args.save_every == 0:
                    logging.info(
                        f"saving checkpoint: ep={epoch:02} it={i:05} ∧={s.status['best_eval']:.4e} λ∧={s.status['pop_best_eval']:.4e} μ={s.status['mean_eval']:.4e}"
                    )
                    save_checkpoint(i)
                    log.to_dataframe().plot(title=title)
                    plt.savefig(f"{exp_dir}/train.png")
                    plt.close()


if __name__ == "__main__":
    main()
