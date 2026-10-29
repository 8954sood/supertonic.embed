"""Validate prepared model contracts without running the browser.

Run after tools/prepare_webgpu.py. This is not a speaker-quality benchmark.
"""
import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1] / 'web/public/models'
EXPECTED = {
    'duration_predictor': {'text_ids', 'style_dp', 'text_mask'},
    'text_encoder': {'text_ids', 'style_ttl', 'text_mask'},
    'vector_estimator': {'noisy_latent', 'text_emb', 'style_ttl', 'latent_mask',
                         'text_mask', 'current_step', 'total_step'},
    'vocoder': {'latent'},
    'wavlm_stats': {'audio'},
}
manifest = json.loads((ROOT / 'manifest.json').read_text())
assert manifest['sampleRate'] == 44100 and manifest['wavlmRate'] == 16000
assert len(manifest['prompts']) == 5
assert set(manifest['presets']) == {f'{sex}{i}' for sex in 'FM' for i in range(1, 6)}
for name, inputs in EXPECTED.items():
    model = onnx.load(str(ROOT / f'{name}.onnx'))
    onnx.checker.check_model(model)
    assert {node.name for node in model.graph.input} == inputs, name
    assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
    if name == 'wavlm_stats':
        assert [out.name for out in model.graph.output] == ['mean', 'std']
    print(f'{name}: valid graph and browser input contract')
print('Prepared asset contracts passed.')
