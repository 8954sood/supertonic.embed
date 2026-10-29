"""Single-reference native MPS extraction with real input gradients and live best saves."""
import argparse
import json
import math
import os
from pathlib import Path
import time


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, default=Path("styles/mps"))
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--model-dir", type=Path, help="Supertonic-2 directory; otherwise use/download its HF snapshot")
    parser.add_argument("--iterations", type=int, default=0, help="0 = until full mean loss <= threshold or Ctrl+C")
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.24)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--preset", choices=["auto"] + [f"{s}{i}" for s in "FM" for i in range(1, 6)], default="auto")
    parser.add_argument("--initial-style", type=Path, help="Start from a saved style (fresh Adam state)")
    parser.add_argument("--skip-duration", action="store_true", help="Save timbre with starting duration style")
    parser.add_argument("--duration-iterations", type=int, default=100)
    parser.add_argument("--duration-rounds", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 0 or args.validation_interval < 1 or args.duration_iterations < 1 or args.duration_rounds < 1:
        parser.error("Invalid iteration settings")
    if not math.isfinite(args.threshold) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Threshold and learning rate must be finite; learning rate must be positive")
    # Set before importing modules that allocate tensors or resolve model paths.
    os.environ["SUPERTONIC_DEVICE"] = args.device
    import torch
    from config import DEVICE, CFG, SEED, SR
    if args.model_dir is None:
        from huggingface_hub import snapshot_download
        args.model_dir = Path(snapshot_download("Supertone/supertonic-2", allow_patterns=["onnx/*", "voice_styles/*.json"]))
    os.environ["SUPERTONIC_ONNX_DIR"] = str(args.model_dir.resolve() / "onnx")
    os.environ["SUPERTONIC_PRESET_DIR"] = str(args.model_dir.resolve() / "voice_styles")
    # Config may already be imported by device validation above; update its paths
    # before dependent modules bind them.
    import config
    config.ONNX_DIR = os.environ["SUPERTONIC_ONNX_DIR"]
    config.PRESET_DIR = os.environ["SUPERTONIC_PRESET_DIR"]
    CFG["duration"]["invert_steps"] = args.duration_iterations
    CFG["duration"]["max_iters"] = args.duration_rounds
    import librosa
    import numpy as np
    import soundfile as sf
    from frozen_tts import FrozenTTS, build_prompts
    from wavlm_loss import load_wavlm, layer_stats, to_wavlm_rate, style_loss
    from style_file import read_style, write_style, preset_paths, preset_name
    from native_optimize import optimize_style

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"Device: {DEVICE}; loading frozen TTS and WavLM layer 4…", flush=True)
    reference, _ = librosa.load(args.audio, sr=16000, mono=True)
    if not 4800 <= len(reference) <= 240000 or not np.isfinite(reference).all() or np.max(np.abs(reference)) < 1e-7:
        raise ValueError("Use finite, non-silent reference audio between 0.3 and 15 seconds")
    tts = FrozenTTS()
    prompts = build_prompts(tts)
    wavlm = load_wavlm(trim=True)
    with torch.no_grad():
        target = layer_stats(wavlm, torch.tensor(reference, device=DEVICE).unsqueeze(0))

    def measure(ttl, dp, index):
        p = prompts[index]
        wav, _ = tts.forward(p.ids, p.mask, ttl, dp, p.latent, p.latent_mask)
        return style_loss(layer_stats(wavlm, to_wavlm_rate(wav)), target)

    if args.initial_style:
        ttl, dp = read_style(args.initial_style)
        preset = "saved style"
    else:
        nearest = None
        for path in preset_paths():
            if args.preset != "auto" and preset_name(path) != args.preset:
                continue
            candidate, duration = read_style(path)
            with torch.no_grad():
                loss = sum(measure(candidate, duration, i).item() for i in range(len(prompts))) / len(prompts)
            print(f"Preset {preset_name(path)}: {loss:.6f}", flush=True)
            if nearest is None or loss < nearest[0]:
                nearest = (loss, preset_name(path), candidate, duration)
        if nearest is None:
            raise ValueError("No matching presets found in model directory")
        _, preset, ttl, dp = nearest

    history = []
    base_metadata = {"method": "pytorch-projected-adam-v1", "device": str(DEVICE),
                     "preset": preset, "seed": SEED, "source_file": str(args.audio),
                     "models": {"supertonic_directory": str(args.model_dir.resolve()),
                                "wavlm_revision": getattr(wavlm.config, "_commit_hash", None)},
                     "objective": "fixed five-prompt mean; WavLM layer 4 mean/std MSE"}

    def save_style(path, value, duration, metadata):
        temp = path.with_suffix(path.suffix + ".tmp")
        write_style(str(temp), value, duration, source_file=str(args.audio), metadata={**base_metadata, **metadata})
        temp.replace(path)

    def on_best(value, metrics):
        save_style(args.out / "best.json", value, dp, {"partial": True, "duration_recovered": False, **metrics})

    def on_step(metrics):
        if metrics["validated"]:
            if len(history) >= 1000:
                del history[1]
            history.append(metrics)
        atomic_json(args.out / "progress.json", {**base_metadata, **metrics})
        print(f"timbre {metrics['step']} | best {metrics['best_loss']:.6f} | "
              f"{metrics['elapsed_seconds']:.1f}s | forwards {metrics['evaluations']} | "
              f"backwards {metrics['backward_calls']}", flush=True)

    print(f"Starting {preset}; real backward gradients, fixed full-objective validation", flush=True)
    best, metrics = optimize_style(ttl, lambda value, index: measure(value, dp, index), len(prompts),
        iterations=args.iterations, threshold=args.threshold, rate=args.learning_rate,
        validation_interval=args.validation_interval, on_step=on_step, on_best=on_best)
    duration_report = {"status": "skipped", "recovered": False}
    duration_complete = False
    if not args.skip_duration and metrics["status"] != "cancelled":
        print("Recovering duration… (best.json remains available)", flush=True)
        atomic_json(args.out / "progress.json", {**base_metadata, **metrics, "stage": "duration"})
        try:
            from invert_dp import recover
            dp, duration_report = recover(reference, best, dp, prompts, tts)
            duration_complete = duration_report["status"] not in ("reference too short", "starting synthesis unmeasurable")
        except KeyboardInterrupt:
            duration_report = {"status": "cancelled", "recovered": False}
            metrics["status"] = "cancelled"
    save_style(args.out / "style.json", best, dp, {**metrics, "partial": not duration_complete,
               "duration_recovered": duration_complete})
    report = {**base_metadata, **metrics, "duration": duration_report, "history": history,
              "total_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
              "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    atomic_json(args.out / "report.json", report)
    atomic_json(args.out / "progress.json", report)
    if metrics["status"] != "cancelled":
        wave, _ = tts.synthesize_at_own_length(prompts[0], best, dp)
        sf.write(args.out / "preview.wav", wave, SR, subtype="PCM_16")
    print(f"{metrics['status']}: {metrics['best_loss']:.6f}; outputs: {args.out}", flush=True)


if __name__ == "__main__":
    main()
