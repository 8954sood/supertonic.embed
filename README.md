# SupertonicTTS Voice Style Extractor

Turn a WAV file into a SupertonicTTS voice style, without the unreleased style
encoder. The output is a style JSON in the same format as the shipped presets:
put it beside `F1.json` … `M5.json` and the model speaks in that voice.

**Requirements:** NVIDIA GPU with CUDA. The default settings need about 10 GB;
`timbre.batch: 1` in `src/config.yaml` runs in 2.5 GB.

## Responsible use

Research code, released for academic purposes. Voice cloning can do serious harm.

- Get explicit consent from anyone whose voice you clone.
- No impersonation, voice phishing, fraud, harassment, defamation, misleading
  content, or defeating voice authentication.
- No identifiable people, public figures included, without permission.
- Label synthetic audio as AI-generated when you distribute it.

The author disclaims all liability for misuse. No style vector of any real
person is released here.

## How it works

```
              ┌───────────────────────────────────────────┐
              │           TTS pipeline (PyTorch)          │
┌───────────┐ │ ┌─────────┐  ┌───────────┐  ┌───────────┐ │ ┌─────────┐
│   style   │→│ │  Text   │→ │  Vector   │→ │  Vocoder  │ │→│ gen WAV │
│  vector   │ │ │ Encoder │  │ Estimator │  │           │ │ └────┬────┘
└─────┬─────┘ │ └─────────┘  └───────────┘  └───────────┘ │      │
      │       └───────────────────────────────────────────┘      │
      │                      ┌────────────┐                      │
      │                      │   WavLM    │◄─────────────────────┘
      │                      │  layer 4   │◄── target WAV
      │                      └─────┬──────┘
      │      gradient              │ loss
      └────────────────────────────┘
```

A style has two parts, recovered differently.

**Timbre** (`style_ttl`, 50×256) is the part that reaches the audio. It starts
at the nearest shipped preset and descends the WavLM layer-4 objective: match
the mean and standard deviation of the layer over time, which makes the loss
independent of what is being said. After every step its 50 rows are put back on
the unit sphere, where every preset row sits. It stops when the loss reaches
0.24, inside the band that two sentences from the *same* preset already span.

**Duration** (`style_dp`, 8×16) only sets how long an utterance runs, which a
time-pooled loss cannot see. So it is matched instead: measure the speaking rate
of the reference on audio alone, then invert the frozen duration predictor until
the synthesis speaks at that rate.

The model is never trained. Only the style vector moves.

## Quick start

```bash
pip install -r requirements.txt
```

Download the model as described in [models/README.md](models/README.md), then:

```bash
python src/run_extract_style_batch.py
```

This reads [src/wavlist.txt](src/wavlist.txt), five example recordings, and
writes `styles/<id>.json` for each plus `styles/extraction_report.json`. About
ten minutes on an RTX 3090.

For your own audio, write a list with one recording per line. `|name` after the
path sets the output filename; otherwise the file's name is used.

```
my_voice.wav
recordings/take_03.wav|alice
```

```bash
python src/run_extract_style_batch.py my_list.txt --out styles
```

| Option | Meaning |
|---|---|
| `--out` | output directory (default `styles`) |
| `--config` | a config file other than `src/config.yaml` |

Everything else is in [src/config.yaml](src/config.yaml). Any sample rate
works. The recording should be mostly speech: a file that is largely silence
describes the room, not a voice.

## Batching

Recordings descend together in one forward pass; each style only reaches its
own output, so the gradient per recording equals a solo run. A recording that
reaches 0.24 leaves the batch at once and a waiting one takes its place.
Measured on an RTX 3090, 5.4 s utterances:

| batch | s / recording / step | peak VRAM |
|---|---|---|
| 1 | 0.465 | 2.5 GB |
| 8 | 0.090 | 9.8 GB |
| 16 | 0.069 | 18.0 GB |

The duration stage runs one recording at a time, since it changes the audio
length.

## Results

147 speakers (103 VCTK, 44 Seed-TTS Eval), 100 held-out sentences each.
Similarity is cosine between speaker embeddings of synthesis and reference.

| | ECAPA ↑ | ResNet ↑ | WavLM-SV ↑ | WER ↓ |
|---|---|---|---|---|
| Nearest preset, no extraction | 0.129 | 0.095 | 0.703 | 6.00% |
| **Extracted** | **0.419** | **0.409** | **0.837** | **5.71%** |

Speaking-rate error against the reference: 9.4% with the preset duration,
2.0% after recovery on the fitted prompts; 11.9% → 8.1% on held-out text.

The 6% WER floor is the frozen model itself, which drops and repeats words
sporadically ([vendor issue](https://github.com/supertone-inc/supertonic/issues/83)).
Extraction adds no intelligibility cost. Seven VCTK references that are mostly
silence or noise were excluded, judged on the reference audio alone.

## Repository

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.yaml                        every tunable value
│   ├── config.py                          reads it
│   ├── text_tokenizer.py                  text → token ids
│   ├── style_file.py                      style JSON in and out, presets, sphere
│   ├── frozen_tts.py                      the released model as frozen torch
│   ├── wavlm_loss.py                      the objective
│   ├── invert_ttl.py                      recover the timbre
│   ├── invert_dp.py                       recover the duration
│   ├── run_extract_style_batch.py         the one thing you run
│   └── wavlist.txt                        default input list
├── examples/
│   ├── 01.wav … 05.wav                    Seed-TTS Eval references
│   └── README.md
└── models/
    └── README.md                          how to download the weights
```

## Licence

MIT. `src/text_tokenizer.py` derives from the SupertonicTTS release, also MIT.
The model weights are not included; Supertone releases them under OpenRAIL-M,
whose use restrictions carry over to anything made with them.

## Citation

```bibtex
@misc{kim2026supertonicembed,
  author        = {Gyeongmin Kim},
  title         = {Extracting Voice Styles from Frozen {TTS} Models via Gradient-Based Inverse Optimization},
  year          = {2026},
  eprint        = {2607.25351},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2607.25351}
}
```
