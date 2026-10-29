# Browser WebGPU extraction (experimental)

This path performs voice-style extraction locally in the browser, including the
WavLM objective and duration fitting. It uses ONNX Runtime Web 1.31.1 with
`['webgpu', 'wasm']`; unsupported operations may execute on the CPU. It requires
WebGPU, a secure context (localhost or HTTPS), and substantial GPU/system memory.
No audio upload or Python inference service is involved.

The original Python extractor remains available. The browser path uses
**preset-guided, blockwise pattern search**, with no automatic differentiation.
It replaces the former full-dimensional SPSA/Adam timbre optimizer. It is still
experimental; CUDA equivalence and general voice quality are not established.
The duration fitter continues to use SPSA with Adam moments.

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
The browser decodes to mono 16 kHz. All ten presets are compared on the **same five-prompt average** used for
acceptance, rather than choosing on one sentence. This costs 50 prompt evaluations
once. The winning preset supplies the fixed duration style during timbre search.
Style rows remain unit length.

**By default the timbre loop has no iteration limit: it runs until loss ≤ 0.24.**
A positive maximum is a bounded experiment. A plateau notice never silently stops
an unlimited run; an unreachable target continues until cancelled.

Each search trial starts at the fully validated best style:

1. Construct a tangent direction toward a released preset, a random block of
   1/5/10 style rows, or a previously accepted direction. Global preset trials
   alternate with block preset trials; random residual directions can leave the
   preset span and eventually reach all rows.
2. Move both ways on the row spheres. Evaluate both candidates on the same
   rotating prompt, with fixed latent noise, against cached incumbent losses.
3. Fully validate the better screening candidate if it improves that prompt.
   Reuse its screening result and evaluate the other four prompts. A periodic
   audit (default every 10 trials) also validates the selected candidate when
   screening fails. Screening is a heuristic and can miss useful candidates.
4. Accept **only** if the five-prompt mean improves. Otherwise retain the best
   style. No Adam state or unmeasured update accumulates in the timbre search.
5. Expand a successful direction's radius, shrink after repeated failures, and
   revisit a fixed radius ladder after 40 unsuccessful trials. These restarts
   change the search scale around the best, not the retained style.

A normal trial costs 2 prompt evaluations if screened out, or 6 with full
validation. Initialization costs 5; preset ranking is additional. Iteration counts
are not comparable to the former SPSA loop: compare prompt evaluations and elapsed
time. Only a fully validated mean can trigger the 0.24 target. This objective is
also different from the original Python single-rotating-prompt best criterion.
Random seed is reproducible within this path, not bitwise identical to NumPy.

Progress reports candidate acceptance/rejection, initial/best loss, evaluation
count, elapsed time, radius, accepted moves and loss reduction per second. The
report retains up to 1,000 full validation history entries (initial plus latest).
The best per-prompt losses belong to the returned best style. A bounded run that
misses the target says so in the main completion message.

The **current best style** download becomes available after search initialization
and refreshes only on fully measured improvements. It remains available after
cancellation. This is a usable style with the starting duration preset, marked
`partial: true` / `duration_recovered: false`, not an optimizer-resume checkpoint.
Refresh or a new run clears this in-memory download; save it before leaving.

Model load/run/readback-copy times, estimator steps and resampling remain in
reports. They measure overlapping wall time, not GPU kernel times or provider
placement. First-call and subsequent-call means are separate; new shapes may
compile later too. Intermediate outputs are still CPU-backed; invariant
estimator feeds are reused within each synthesis and released on failure.

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

## Reproduce the algorithm comparison

With prepared models and the development server running:

```sh
cp examples/01.wav web/public/benchmark-reference.wav
```

Open `/benchmark.html` on the local server and run the comparison. Both algorithms
start at M1 with identical fixed prompt/noise settings, independent of the new
preset-ranking procedure. The budget counts actual TTS + WavLM prompt evaluations,
including optimizer initialization. Only completed full validations can improve
the recorded best. Initialization/download time and preset ranking are excluded;
wall time includes incomplete work at budget exhaustion. Reverse the algorithm
order to check warm-cache effects. This single CC0 reference is a smoke comparison,
not a general quality/convergence benchmark. Duration fitting is excluded.
