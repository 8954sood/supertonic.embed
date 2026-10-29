# Browser WebGPU extraction (experimental)

This path performs voice-style extraction locally in the browser, including the
WavLM objective and duration fitting. It uses ONNX Runtime Web 1.31.1 with
`['webgpu', 'wasm']`; unsupported operations may execute on the CPU. It requires
WebGPU, a secure context (localhost or HTTPS), and substantial GPU/system memory.
No audio upload or Python inference service is involved.

The original Python extractor remains available. Its backpropagation cannot be
replaced by selecting an ONNX execution provider: this implementation instead
uses two-sided SPSA estimates and Adam moments to optimize the input styles
without automatic differentiation. This is an experimental alternative algorithm,
not a numerically equivalent port. Convergence, speaker similarity, runtime and
memory against the CUDA implementation have **not** been established. In
particular, 12,800-dimensional timbre SPSA can converge very slowly or stall.

## Prepare once

Use Python 3.12+ and Node 20.19+ (or 22.12+).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-webgpu.txt
python tools/prepare_webgpu.py
npm --prefix web ci
npm --prefix web run dev
```

Open the local URL printed by the development server. Preparation downloads the
Supertonic-2 ONNX models/presets and WavLM-Large. Downloads require several GB of
disk space including caches; the browser asset set is about 500 MB. Preparation
runs on CPU, with no NVIDIA GPU required. Existing Supertonic-2 files can be used:

```sh
python tools/prepare_webgpu.py --model-dir /path/to/supertonic-2
```

That directory must contain `onnx/` and `voice_styles/`. Do not substitute
Supertonic-3 or optimized variants with different inputs. Weights are not tracked
in Git. Their existing licenses still apply (Supertonic OpenRAIL-M, WavLM MIT).

Preparation exports only the first four WavLM blocks. For WavLM-Large's stable
layer normalization it removes the **final encoder** layer norm after truncation:
the original `hidden_states[4]` precedes that norm. It verifies the trimmed
objective against the full PyTorch model, then checks ONNX/PyTorch agreement at
two audio lengths. `manifest.json` is written only after these checks pass.

## Extract

Choose an authorized reference recording of 0.3–15 seconds, mostly clean speech.
The browser decodes to mono 16 kHz. The ten presets are compared against WavLM
layer-four statistics; the nearest initializes the style. Timbre loss is the sum
of mean and sample-standard-deviation MSEs, as in Python. All style rows are
projected to unit length after each update.

**By default the timbre loop has no iteration limit: it runs until loss ≤ 0.24.**
Set the maximum iterations field to 0 for this behavior. A positive value is an
optional bounded experiment; its output may not reach the target. Extraction runs in a dedicated worker so model initialization does not block the
interface. Cancellation terminates the worker and releases its resources. An unreachable threshold or stalled SPSA run
continues until cancelled. Finishing is not guaranteed.

The two SPSA probes share a prompt and fixed noise, with prompts rotated across
steps. Best-style acceptance and stopping use the **average over all five
prompts**, so losses from different steps are comparable. Python accepts the best
single rotated-prompt loss instead, so stopping criteria are intentionally more
conservative here. Random seed is reproducible within this browser path, but
latents are not bitwise identical to NumPy's generator.

Duration recovery fits the predictor with SPSA and a prior toward the preset,
then evaluates generated speaking rates. It retains the best measured style,
with up to eight outer iterations and a default 100 SPSA steps per fit. The
browser RMS/peak detector approximates librosa/scipy; it is not bitwise equivalent.
Reports distinguish convergence, unmeasurable audio, lack of improvement and the
iteration limit. Duration fitting retains its own bounds, like the Python path.

Save the style JSON (`style_ttl [1,50,256]`, `style_dp [1,8,16]`), extraction report,
and generated preview WAV. Reports record method, seed, budgets, losses and rate
history; no quality claims are inferred from simply completing a run.

## Build and checks

```sh
npm --prefix web test
python tests/verify_webgpu_assets.py
npm --prefix web run build
```

Host `web/dist/` with its prepared `models/` directory over HTTPS. The model folder
URL can point to a different static host if it permits CORS. Keep ONNX files,
manifest and WASM assets reachable. This is a static application; there is no
backend to receive recordings.

Numerical tests cover constrained optimization, unlimited threshold stopping,
cancellation, style-file compatibility, interpolation, seeded noise, rate
measurement and WAV output. They do not establish real-speaker extraction quality.

Official execution-provider documentation:
https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

See [recorded implementation checks](../docs/WEBGPU_VALIDATION.md) for the actual
browser smoke run and its limits. Duration predictor calls are intentionally
sequential because overlapping asyncify session runs stalled the initial browser
implementation.
