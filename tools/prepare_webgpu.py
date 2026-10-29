"""Prepare local-only browser assets; no CUDA required (Python runs on CPU).

Run with the repository Python dependencies plus huggingface_hub and onnxscript.
Only preparation uses Python. Extraction then runs entirely in the browser.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, help='Existing supertonic-2 directory containing onnx/ and voice_styles/')
    ap.add_argument('--out', type=Path, default=ROOT / 'web/public/models')
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    import numpy as np
    import onnx
    import torch
    from transformers import WavLMModel

    model_dir = args.model_dir or Path(snapshot_download('Supertone/supertonic-2',
        allow_patterns=['onnx/*', 'voice_styles/*.json']))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    for name in ('duration_predictor', 'text_encoder', 'vector_estimator', 'vocoder'):
        source = model_dir / 'onnx' / f'{name}.onnx'
        # Inline external data so browser requests do not depend on server paths.
        model = onnx.load(str(source), load_external_data=True)
        onnx.save_model(model, str(out / f'{name}.onnx'), save_as_external_data=False)
    shutil.copyfile(model_dir / 'onnx/tts.json', out / 'tts.json')
    for source in sorted((model_dir / 'voice_styles').glob('[FM]*.json')):
        shutil.copyfile(source, out / source.name)
    sys.path.insert(0, str(ROOT / 'src'))
    from text_tokenizer import TextTokenizer
    import yaml
    cfg = yaml.safe_load((ROOT / 'src/config.yaml').read_text())
    tokenizer = TextTokenizer(model_dir / 'onnx/unicode_indexer.json')
    prompts = [{'text': text, 'ids': tokenizer(text)[0][0].tolist()} for text in cfg['prompts']]
    manifest = {'version': 1, 'sampleRate': 44100, 'wavlmRate': 16000,
                'steps': cfg['model']['total_step'], 'speed': cfg['model']['speed'],
                'seed': cfg['model']['seed'], 'prompts': prompts,
                'presets': [p.stem for p in sorted(out.glob('[FM]*.json'))]}
    if len(manifest['presets']) != 10:
        raise RuntimeError('Expected all ten F1-F5 / M1-M5 presets')

    class Statistics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wavlm = WavLMModel.from_pretrained('microsoft/wavlm-large').eval()
            self.probe = torch.randn(1, 32000)
            with torch.no_grad():
                h = self.wavlm(self.probe, output_hidden_states=True).hidden_states[4]
                self.expected = (h.mean(dim=1), h.std(dim=1))
            # A stable-layer-norm encoder normalizes only its FINAL layer.
            # The original hidden_states[4] precedes that final normalization.
            if self.wavlm.config.do_stable_layer_norm:
                self.wavlm.encoder.layer_norm = torch.nn.Identity()
            # hidden_states[4] is the output of the fourth transformer block.
            # Trim unused blocks before export to reduce browser memory/download.
            self.wavlm.encoder.layers = self.wavlm.encoder.layers[:4]
            self.wavlm.config.num_hidden_layers = 4
            for param in self.parameters():
                param.requires_grad_(False)

        def forward(self, audio):
            h = self.wavlm(audio).last_hidden_state
            mean = h.mean(dim=1)
            centered = h - mean.unsqueeze(1)
            # Original layer_stats uses sample std (torch.std correction=1).
            std = torch.sqrt((centered * centered).sum(dim=1) / (h.shape[1] - 1))
            return mean, std

    stats = Statistics().eval()
    with torch.no_grad():
        for actual, expected in zip(stats(stats.probe), stats.expected, strict=True):
            torch.testing.assert_close(actual, expected)
    example = torch.randn(1, 16000 * 2)
    path = out / 'wavlm_stats.onnx'
    with torch.no_grad():
        torch.onnx.export(stats, (example,), str(path), input_names=['audio'],
                          output_names=['mean', 'std'], opset_version=17, dynamo=False,
                          dynamic_axes={'audio': {1: 'samples'}})
    # Check numerical agreement at two lengths, including a non-export length.
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for samples in (32000, 48000):
        audio = torch.randn(1, samples)
        with torch.no_grad():
            expected = stats(audio)
        actual = session.run(None, {'audio': audio.numpy()})
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b.numpy(), rtol=3e-3, atol=3e-4)
    # Write the readiness marker last, after every export and comparison succeeds.
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared and checked browser assets: {out}')


if __name__ == '__main__':
    main()
