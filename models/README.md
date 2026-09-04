# Model files

Not tracked here. The extractor needs the released SupertonicTTS model, which
Supertone distributes under the OpenRAIL-M licence.

Download the `onnx/` and `voice_styles/` folders from
[Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2) and put
them beside this file, so the tree looks like:

```
models/
├── onnx/
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── vector_estimator.onnx
│   ├── vocoder.onnx
│   ├── tts.json
│   └── unicode_indexer.json
└── voice_styles/
    ├── F1.json … F5.json
    └── M1.json … M5.json
```

Both paths are set in `src/config.yaml`, so they can live elsewhere.

WavLM-Large is downloaded automatically from HuggingFace the first time the
extractor runs (`microsoft/wavlm-large`, about 1.2 GB), and cached.
