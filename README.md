# SupertonicTTS Voice Style Extractor

Recover a voice style for SupertonicTTS from a plain WAV file, without the
official style encoder, which was never released.

Give it a recording of a few seconds and it writes a style JSON in the same
format as the presets the model ships with. Drop that file in beside F1..F5 and
M1..M5 and the model speaks in that voice.

**Requirements:** an NVIDIA GPU and CUDA. The default settings optimize eight
recordings at once and need about 10 GB; set `timbre.batch: 1` in
`src/config.yaml` to run in 2.5 GB.

## Responsible Use

**This is research code, released for academic purposes.** Voice cloning can be
misused for serious harm. By using this repository you agree to the following.

- **Obtain explicit consent** from anyone whose voice you clone. Cloning a real
  person's voice without permission may be illegal where you live.
- **Do not use this tool** for non-consensual impersonation, voice phishing,
  fraud, harassment, defamation, misleading political or commercial content, or
  to defeat voice authentication.
- **Do not target identifiable people**, public figures included, without their
  explicit permission.
- **Disclose synthetic audio** as AI-generated when you distribute it, ideally
  with a watermark or provenance metadata such as C2PA.

The author disclaims all liability for misuse. The evaluation behind the numbers
below uses VCTK (CC BY 4.0) and the Common Voice subset of Seed-TTS Eval (CC0),
targets no identifiable individual, and releases no extracted style vector for
any real person. If you see this code being misused, please open an issue.

## How it works

```
              ┌───────────────────────────────────────────┐
              │           TTS pipeline (PyTorch)          │
┌───────────┐ │ ┌─────────┐  ┌───────────┐  ┌───────────┐ │ ┌─────────┐
│   style   │→│ │  Text   │→ │  Vector   │→ │  Vocoder  │ │→│ gen WAV │
│  vector   │ │ │ Encoder │  │ Estimator │  │           │ │ └────┬────┘
└─────┬─────┘ │ └─────────┘  └───────────┘  └───────────┘ │      │
      │       └───────────────────────────────────────────┘      │
      │                                                          │
      │                      ┌────────────┐                      │
      │                      │   WavLM    │◄─────────────────────┘
      │                      │  Layer 4   │
      │                      │  (speaker  │◄── target WAV
      │                      │ identity)  │
      │                      └─────┬──────┘
      │                            │
      │      gradient              │ loss
      └────────────────────────────┘
      "update style to be more similar"
```

A style has two halves, and they are recovered by different means because they
carry different things.

**Timbre** (`style_ttl`, 50x256) is the only half that reaches the audio. It is
recovered by gradient descent: start at the closest shipped preset, synthesize,
compare against the target with time-pooled WavLM layer-4 statistics, step
downhill, and put all 50 token rows back on the unit sphere. Stop when the loss
reaches 0.24.

**Duration** (`style_dp`, 8x16) only tells the duration predictor how long an
utterance runs, so WavLM cannot see it: a time-pooled objective discards exactly
the information tempo lives in. It is matched instead. The speaking rate of the
reference is measured on audio alone, and the duration style is inverted through
the frozen duration predictor until the synthesis speaks at the same rate.

Nothing in the model is trained. Only the style vector moves.

### Why the unit sphere

Every token row of every shipped preset has unit L2 norm, to float precision.
That is the surface the withheld encoder writes onto. Free gradient descent
leaves it, and the further the vector drifts the worse the synthesis gets.
Renormalizing each row after every step keeps the iterate on the manifold the
model was trained to consume.

### Why 0.24

Synthesizing two different sentences from the *same* preset still leaves a
residual loss, caused by the content rather than by any difference of speaker.
Across the ten presets that floor spans 0.15 to 0.41 at layer 4. Anything inside
that band is already at same-voice distance, so 0.24 is a stopping point, not a
convergence point. Over 147 speakers every run reached it, in a median of 460
steps.

The floor is layer-specific, so changing the layer invalidates the threshold.

## Quick start

```bash
pip install -r requirements.txt
```

Download the model as described in [models/README.md](models/README.md), then:

```bash
python src/run_extract_style_ttl_dp_batch.py
```

That reads [src/wavlist.txt](src/wavlist.txt), which points at the five example
recordings in [examples/](examples/), and writes `styles/<id>.json` for each,
plus `styles/extraction_report.json` with the loss, step count, chosen preset
and speaking-rate fit of every recording. The five examples take about ten
minutes on an RTX 3090.

To run it on your own audio, put one recording per line in a list of your own.
A line may name an id after a pipe; otherwise the file's own name is used. Blank
lines and `#` comments are skipped.

```
my_voice.wav
recordings/take_03.wav|alice
```

```bash
python src/run_extract_style_ttl_dp_batch.py my_list.txt --out styles
```

| Option | Meaning |
|---|---|
| `--out` | where the style JSONs go (default `styles`) |
| `--config` | use a config file other than `src/config.yaml` |
| `--timbre-only` | skip the duration stage and leave `style_dp` at the preset's |

Everything else is set in [src/config.yaml](src/config.yaml): the WavLM layer,
the stopping loss, the learning rate, how many recordings descend together, the
sampling steps, and the prompts the objective rotates through.

Any sample rate works; audio is resampled on the way in. The recording has to be
mostly speech. The objective pools over time, so a file that is largely silence
gives a target describing the room rather than a voice.

## Extracting many voices at once

Several style vectors descend in a single forward pass. Each one reaches only
its own output, so stacking B of them and summing their losses gives every
vector exactly the gradient it would have received alone; summing rather than
averaging is what keeps the effective learning rate per recording the same as a
solo run. All four ONNX graphs take a dynamic batch dimension, and every
recording uses the same prompt on a given step, so nothing needs padding.

Measured on an RTX 3090 with 24 GB, on 5.4 s utterances:

| batch | s/recording/step | speedup | peak VRAM |
|---|---|---|---|
| 1 | 0.465 | 1.0x | 2.5 GB |
| 4 | 0.141 | 3.3x | 5.6 GB |
| 8 | 0.090 | 5.2x | 9.8 GB |
| 16 | 0.069 | 6.7x | 18.0 GB |
| 24 | 0.337 | 1.4x | 26.1 GB, over the card, thrashes |

About 1.03 GB per extra recording, so 16 is the practical maximum on 24 GB. Set
it with `timbre.batch` in the config.

Two rules the batching respects. A recording that reaches the threshold leaves
the batch immediately, because further descent walks its vector into territory
the frozen model never saw. And a waiting recording takes its place at once, so
the batch stays full.

The duration stage runs one recording at a time, because the duration style sets
the length of the audio and the recordings no longer share a tensor shape.

## Performance

147 speakers, 103 from VCTK and 44 from the Common Voice subset of Seed-TTS
Eval, each evaluated on 100 held-out sentences, 113,907 words per system.
Similarity is cosine between speaker embeddings of the synthesis and the
reference.

| | SIM ECAPA ↑ | SIM ResNet ↑ | SIM WavLM-SV ↑ | WER ↓ |
|---|---|---|---|---|
| Nearest preset, no extraction | 0.129 | 0.095 | 0.703 | 6.00% |
| Timbre extracted | 0.428 | 0.416 | 0.840 | 5.72% |
| **Timbre and duration extracted** | **0.419** | **0.409** | **0.837** | **5.71%** |

Extraction more than triples speaker similarity and costs no intelligibility:
the extracted voices transcribe slightly better than the model's own presets do.
That 6% floor is the frozen model dropping and repeating words on its own, not a
cost of the method. See the caveat below.

On the ten short prompts, five of which the optimization saw:

| prompts | SIM ECAPA ↑ | WER ↓ |
|---|---|---|
| Five fitted | 0.483 | 1.13% |
| Five held out | 0.473 | 1.62% |

Eleven thousandths of similarity is what generalization to unseen text costs.

Duration recovery, as the distance between the synthesized and the reference
speaking rate:

| duration style | fitted prompts | held-out prompts |
|---|---|---|
| Preset | 9.4% | 11.9% |
| Recovered | **2.0%** | **8.1%** |

Similarity does not move, because `style_dp` never reaches the acoustic path. On
held-out text the fitted tempo transfers about halfway: the rate detector counts
syllable nuclei per second, which also depends on how dense the sentence is.

A reference was used only if its own audio carries enough speech to describe a
voice: a measured rate of at least 2.2 nuclei per second over the whole file,
and crude signal-to-noise of at least 22 dB. That drops 7 of the 154 VCTK
references, all of them mostly silence or mostly noise. The criterion looks at
the reference alone, never at what the extraction produced.

### A caveat on WER

SupertonicTTS drops and repeats words sporadically on its own, which the vendor
has [acknowledged](https://github.com/supertone-inc/supertonic/issues/83) and
addressed in Supertonic 3. The preset row above measures exactly that: the
frozen model, its own shipped style vectors, 6.00% word error rate over 14,700
sentences. Any single-clip word error rate from this model is unreliable, and
the same preset and sentence can swing by fifteen points across noise seeds,
which is why the figures above are pooled over 100 sentences per speaker.

## What is in the repository

```
.
├── README.md
├── LICENSE                              MIT
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.yaml                      every tunable value
│   ├── config.py                        reads it, plus the structural constants
│   ├── text_tokenizer.py                text to the token ids the model expects
│   ├── style_file.py                    the style JSON: read, write, presets, sphere
│   ├── frozen_tts.py                    the released ONNX as frozen differentiable torch
│   ├── wavlm_loss.py                    the WavLM layer-4 objective
│   ├── invert_ttl.py                    recover the timbre
│   ├── invert_dp.py                     recover the duration
│   ├── run_extract_style_ttl_dp_batch.py  the one thing you run
│   └── wavlist.txt                      the default input list
├── examples/
│   ├── 21.wav
│   ├── 22.wav
│   ├── 23.wav                           five Seed-TTS Eval references
│   ├── 24.wav                           (Common Voice, CC0)
│   ├── 25.wav
│   └── README.md                        where each one came from
└── models/
    └── README.md                        how to download the model; the weights are not here
```

That is the whole repository. The model weights, WavLM, and anything the
extractor writes are downloaded or created locally and stay out of it.

## Models used

| Model | Role |
|---|---|
| duration_predictor | how long each utterance runs (SupertonicTTS) |
| text_encoder | text to embeddings (SupertonicTTS) |
| vector_estimator | flow-matching denoiser (SupertonicTTS) |
| vocoder | latent to waveform (SupertonicTTS) |
| WavLM-Large | the objective, layer 4 (microsoft/wavlm-large) |

## Technical notes

### ONNX to PyTorch

The released ONNX modules are converted so gradients can reach the style vector:
onnxslim for cleanup, opset forced to 17 for onnx2torch, and trailing empty Clip
inputs dropped. Every converted parameter is frozen.

### WavLM layer-4 feature matching

[Chiu et al. (2025)](https://arxiv.org/abs/2501.05310) report that for
WavLM-Large, speaker-identification accuracy peaks at layer 4, with gender at
layer 2 and pitch at layer 1. Their layer 0 is the CNN encoder output, which
matches HuggingFace's `hidden_states` indexing, so `hidden_states[4]` is their
layer 4.

The objective compares the mean and standard deviation over time of that layer.
Collapsing the time axis is what makes it content-independent: the two clips may
say different words. It is also the statistics pooling x-vector speaker
embeddings use, which is why matching it transfers to speaker-verification
metrics it never saw.

### Style space

- `style_ttl` [1, 50, 256], 12,800 numbers, the timbre. It enters the text
  encoder and the vector estimator, so it is the only part that reaches the
  audio.
- `style_dp` [1, 8, 16], 128 numbers, the duration. It enters the duration
  predictor and nothing else, so it sets how long an utterance runs and never
  touches the timbre.

Every row of both is kept at unit L2 norm, matching the shipped presets. Male
and female voices occupy different regions of the space, and the extractor
starts from the nearest preset.

## Licence

MIT, see [LICENSE](LICENSE). `src/text_tokenizer.py` is derived from the
SupertonicTTS release, also MIT.

The model weights are not distributed here. Supertone releases them under
OpenRAIL-M, which carries use restrictions, and an extracted style vector is
meaningless without them, so those restrictions apply to what you do with it.

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

Preprint: https://arxiv.org/abs/2607.25351

## Questions

Open an issue.
