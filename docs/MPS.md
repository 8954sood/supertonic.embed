# Native Mac MPS extraction

This is a separate local Python path using Apple GPU acceleration and actual
backpropagation through the frozen TTS and WavLM models. The browser WebGPU path
remains available. This path is experimental; faster target attainment, speaker
quality and CUDA equivalence are not established across references/devices.

## Start the local web interface

Use Python 3.12 on an Apple Silicon Mac:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mps.txt
.venv/bin/python src/serve_mps.py
```

Open `http://127.0.0.1:8765/`. Choose a reference recording between 0.3 and
15 seconds, then start extraction. The first run downloads/caches Supertonic-2
and WavLM-Large if needed. Existing Hugging Face caches are reused. No separate
WebGPU asset export is required. Network downloads use normal TLS verification.

The interface sends the recording to **this Mac's loopback-only Python server**,
not an external inference service. Each run is written under `styles/mps-web/`:
the input audio, run log, live progress, best style, final style/report and preview.
These local files persist after cancellation/restart and are excluded from Git.
The server permits one extraction at a time; stopping it interrupts an active run.

The **target loss** field accepts any finite nonnegative value (default 0.24),
including lower targets such as 0.1 or 0.05. Maximum timbre iterations **0** means
continue until the fixed five-prompt mean reaches the selected target or the user
cancels. A plateau notice never silently stops it. The target is fixed for each
run, displayed in progress and recorded in saved metadata/reports. Lower loss
does not by itself establish intelligibility or voice quality.
The best-style download is available before duration recovery and remains after
cancellation. Bounded experiments explicitly report an iteration limit when the
target is missed. Skipping duration retains the starting duration style and marks
the output partial. The server is for local use and binds only to `127.0.0.1`.

## Command line

```sh
.venv/bin/python src/run_extract_style_mps.py examples/01.wav --out styles/my-voice
```

Use `--threshold 0.05` to set a custom target on the command line.

Useful bounded experiments:

```sh
.venv/bin/python src/run_extract_style_mps.py examples/01.wav \
  --iterations 30 --preset M1 --skip-duration --out styles/mps-test

.venv/bin/python src/run_extract_style_mps.py examples/01.wav \
  --initial-style styles/mps-test/best.json --out styles/mps-continue
```

Starting from a saved style creates fresh Adam state; it is **not exact optimizer
resume**. The normal start ranks all ten presets on the same five-prompt objective
as final acceptance. `--preset M1` skips ranking other presets for experiments.
`--model-dir /path/to/supertonic-2` uses an existing directory containing `onnx/`
and `voice_styles/`. `--device cpu`/`cuda` can exercise the same native path on
other hardware. MPS is explicitly requested and errors if unavailable; it does
not silently switch the entire job to CPU. Unsupported MPS operations error by
default; `PYTORCH_ENABLE_MPS_FALLBACK=1` is an explicit opt-in, recorded in reports.

## Optimizer and objective

The native loop uses a rotating prompt per gradient update, projected Adam,
gradient-norm clipping and full validation every five updates (plus initialization
and the final bounded update). Only the fully validated mean can update the saved
best or stop at the target. A single training-prompt loss is never a success
criterion. Default learning rate is **0.002**, configurable using `--learning-rate`.
It improved the bounded example below; it is not a universal optimum.

After 20 validation rounds without improvement, the working style returns to the
measured best, Adam state clears, and the learning rate halves (floor 1% of the
initial rate). The run continues. Every improved best is saved atomically.
Ctrl+C retains the last completed full validation; partially evaluated candidates
cannot replace it. `best.json` uses the initial duration style. After optional
duration recovery, `style.json`, `report.json` and `preview.wav` are written.
Duration fitting still uses the original sequential differentiable predictor.

WavLM is trimmed to four encoder blocks to avoid unused work/memory, preserving
the original layer-four hidden state before the stable encoder's final norm.
Model parameters stay frozen while gradients reach the input timbre style.
The original CUDA batch entry point keeps its existing default device selection;
the native runner selects MPS explicitly.

## Validation — 2026-10-30

Test environment: Apple M4 Pro, 48 GiB unified memory, macOS ARM64, Python 3.12.6,
PyTorch 2.15.0, Transformers 4.57.6, onnx2torch 1.5.15, onnxslim 0.1.98.
MPS CPU-fallback environment variable was **not enabled**.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python tools/verify_mps.py
```

Nine CPU unit tests cover true-gradient optimization, fixed-mean acceptance,
bounded final validation, recovery, cancellation, non-finite rejection, WavLM
truncation under both layer-norm modes, loopback/origin checks and JSON downloads.
On this machine, all four converted TTS modules agree with ONNX CPU at matching
inputs (`rtol=3e-3`, `atol=3e-4`). Trimmed WavLM MPS/CPU statistics also agree at
those tolerances. The end-to-end TTS → WavLM backward reaches the style with a
nonzero, finite gradient.

Bounded timbre-only runs on repository CC0 `examples/01.wav`, seed 42, M1 start:

| Learning rate | Updates | Initial full mean | Best full mean | Optimization seconds | Forward / backward calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0002 | 10 | 0.92428514 | 0.88834549 | 15.344 | 25 / 10 |
| 0.002 | 30 | 0.92428514 | 0.51371615 | 38.110 | 65 / 30 |

The second run took 44.240 seconds including initialization, M1 measurement and
output work before preview synthesis. Its ten-update full loss was 0.760274.
Optimization time includes initial full validation; model setup, preset ranking,
duration fitting and final preview are outside that timer. Counters separate
forward and backward work; a backward is not equivalent in cost to one inference.

These runs use the original Python NumPy noise generator and librosa reference
resampling. The WebGPU benchmark uses different latent noise and browser audio
decoding, so its loss/time figures are **not a controlled native/WebGPU speedup
comparison**. Target attainment at 0.24, held-out speech quality, general convergence
and peak memory remain unverified. No multiplier speed claim is made.

The exact TTS ONNX and reference hashes are recorded in
`docs/WEBGPU_VALIDATION.md`; native inference uses the same released Supertonic-2
graphs before conversion. Browser WavLM exports are not used for native gradients.

The local web interface also completed the full path with one timbre update and
one update per duration fit: auto selection picked M1, full mean improved to
0.89106133 and duration recovery finished at `noise floor`. Final/partial JSON
read-back confirmed expected shapes, finite values and unit row norms within
1e-6; preview is a valid 5.433-second mono PCM16 44.1 kHz WAV. The interface clearly
reported the iteration limit rather than target success.

A separate unlimited UI run was cancelled during the eleventh backward/update.
The best previously validated mean (0.76027418 at update 10) remained downloadable,
and the downloaded JSON was read back with its partial/duration-incomplete flags.
The final report recorded cancellation; Start became available again. The server
remained running for a subsequent user extraction.
