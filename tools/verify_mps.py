"""Check native MPS TTS, WavLM statistics and backward compatibility on prepared weights."""
import argparse
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.model_dir is None:
        from huggingface_hub import snapshot_download
        args.model_dir = Path(snapshot_download("Supertone/supertonic-2", allow_patterns=["onnx/*", "voice_styles/*.json"]))
    os.environ["SUPERTONIC_DEVICE"] = "mps"
    os.environ["SUPERTONIC_ONNX_DIR"] = str(args.model_dir.resolve() / "onnx")
    os.environ["SUPERTONIC_PRESET_DIR"] = str(args.model_dir.resolve() / "voice_styles")
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import onnxruntime as ort
    import torch
    from frozen_tts import FrozenTTS, build_prompts
    from style_file import read_style, preset_path
    from wavlm_loss import load_wavlm, layer_stats, to_wavlm_rate
    tts = FrozenTTS()
    prompt = build_prompts(tts)[0]
    ttl, dp = read_style(preset_path("M1"))

    def cpu(tensor):
        return tensor.detach().cpu().numpy()

    def compare(name, feeds, expected):
        session = ort.InferenceSession(str(args.model_dir / "onnx" / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        result = session.run(None, {key: cpu(feeds[key]) for key in names})[0]
        np.testing.assert_allclose(cpu(expected), result, rtol=3e-3, atol=3e-4)
        print(f"{name}: native MPS/ONNX CPU agreement", flush=True)

    with torch.no_grad():
        duration = tts.duration(prompt.ids, dp, prompt.mask)
        compare("duration_predictor", {"text_ids": prompt.ids, "style_dp": dp, "text_mask": prompt.mask}, duration)
        text = tts.text_encoder(prompt.ids, ttl, prompt.mask)
        compare("text_encoder", {"text_ids": prompt.ids, "style_ttl": ttl, "text_mask": prompt.mask}, text)
        now, total = torch.tensor([0.], device="mps"), torch.tensor([5.], device="mps")
        latent = tts.vector_estimator(prompt.latent, text, ttl, prompt.latent_mask, prompt.mask, now, total)
        compare("vector_estimator", {"noisy_latent": prompt.latent, "text_emb": text, "style_ttl": ttl,
            "latent_mask": prompt.latent_mask, "text_mask": prompt.mask, "current_step": now, "total_step": total}, latent)
        wave = tts.vocoder(latent)
        compare("vocoder", {"latent": latent}, wave)
    model = load_wavlm(trim=True)
    audio = torch.from_numpy(np.random.default_rng(42).normal(0, 0.1, (1, 32000)).astype(np.float32)).to("mps")
    with torch.no_grad():
        mps = layer_stats(model, audio)
    model.cpu()
    with torch.no_grad():
        expected = layer_stats(model, audio.cpu())
    for a, b in zip(mps, expected, strict=True):
        np.testing.assert_allclose(cpu(a), cpu(b), rtol=3e-3, atol=3e-4)
    print("WavLM layer 4: MPS/CPU statistics agreement", flush=True)
    model.to("mps")
    ttl = ttl.detach().requires_grad_(True)
    wave, _ = tts.forward(prompt.ids, prompt.mask, ttl, dp, prompt.latent, prompt.latent_mask)
    stats = layer_stats(model, to_wavlm_rate(wave))
    loss = sum(s.square().mean() for s in stats)
    loss.backward()
    assert ttl.grad is not None and torch.isfinite(ttl.grad).all().item() and ttl.grad.norm().item() > 0
    print(f"End-to-end native MPS backward passed: style gradient norm {ttl.grad.norm().item():.6f}", flush=True)


if __name__ == "__main__":
    main()
