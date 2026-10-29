# WebGPU extraction TODO

Last updated: 2026-10-29 (Asia/Seoul)

## Current state

- Browser-only extraction is implemented: reference audio → nearest preset →
  timbre optimization → duration recovery → style JSON, report and preview WAV.
- Inference uses ONNX Runtime Web 1.31.1 with WebGPU and WASM fallback.
- Timbre optimization is **SPSA with Adam moments**, not the original PyTorch
  backpropagation. The default is unlimited iterations until loss ≤ 0.24, with
  manual cancellation. Attaining that threshold is not guaranteed.
- Best loss / stopping uses the average of all five fixed prompts. The original
  Python code tracks losses from individual rotating prompts; the numbers are
  therefore not directly comparable as stopping criteria.
- A duration-stage browser stall was resolved by serializing duration predictor
  calls. Do not overlap runs on the current asyncify session.
- Baseline validation is recorded in `docs/WEBGPU_VALIDATION.md`.
- User-observed slow convergence: **478 iterations → best loss 0.88410**,
  then **741 iterations → best loss 0.87335**. The additional 263 iterations
  improved best loss by 0.01075 (about 1.2%). These UI observations are not a
  controlled convergence benchmark and do not predict time to loss 0.24. No
  optimizer improvements addressing this slow convergence have been applied yet.
- The working style continues from every Adam update, including unsuccessful
  updates; the displayed best loss belongs to a separately retained best style.
  Current loss may therefore be worse than the displayed best loss.

## Priority 1: preserve work and measure convergence

- [ ] Export the best style and optimizer state periodically, before target loss
  is reached; support downloading the current best without waiting for duration
  recovery. Current cancellation discards the in-memory run.
- [ ] Support resuming checkpoints, including current/best style, Adam moments,
  iteration, RNG state, prompt/noise settings, target objective and model version.
- [ ] Record initial, current and best loss, accepted improvements, objective-call
  count, elapsed time and per-prompt loss. Show a loss history and plateau status.
  A plateau notification must not silently stop the requested unlimited run.
- [ ] Build a repeatable, consented/CC0 reference-audio benchmark using fixed
  model revisions and seeds. Compare equal wall time and equal objective-call
  budgets, not iteration counts alone.

## Priority 2: improve timbre optimization

The following are hypotheses to evaluate, not established fixes.

- [ ] Quantify the variance of the current two-probe gradient estimate across
  independent directions at the same style. There are 12,800 timbre parameters;
  a single random direction can provide a noisy estimate.
- [ ] Compare averaging multiple perturbation directions per update, including
  its extra inference cost. Keep plus/minus probes on identical prompt/noise.
- [ ] Compare block/row-wise or structured low-dimensional perturbations, with
  schedules that eventually permit changing all style coordinates.
- [ ] Sweep perturbation magnitude and learning rate. Check tangent-space
  perturbations/projection so sphere constraints do not obscure the estimate.
- [ ] Evaluate recovery from unsuccessful updates: restart from the measured
  best style, reset/damp Adam moments after sustained regression, or perform a
  measured step-size search. Track the cost of all extra evaluations.
- [ ] Evaluate preset mixtures or several starting presets rather than only the
  single-prompt nearest preset; select with the same validation objective.
- [ ] Reduce the current five-prompt validation cost per update only if best-loss
  acceptance and target stopping remain comparable and correctly measured.
- [ ] Research a true-gradient browser path: exported backward graphs or an
  autodiff-capable WebGPU implementation of TTS + WavLM. Establish operator,
  memory and numerical feasibility before claiming parity with CUDA/Adam.
- [ ] Compare speaker similarity, intelligibility and held-out text, in addition
  to optimization loss. Do not infer voice quality from loss reduction alone.

## Performance diagnosis and next experiments

The user also reports slow execution. The following operation counts are derived
from the current code; their wall-time contributions have not yet been profiled.

- Each ordinary timbre update performs two SPSA probe evaluations plus five
  prompt evaluations for acceptance/stopping: **seven TTS + WavLM evaluations**.
- Each evaluation runs one text encoder, five vector-estimator steps, one
  vocoder and one WavLM statistics graph: **eight session runs**, or approximately
  **56 model session runs per timbre update**, excluding initialization and
  duration recovery. Early threshold stopping may perform fewer updates.
- Each session output is read with `getData()` and copied into a Float32Array,
  then provided as CPU-backed input to the next session. GPU-resident chaining
  has not been implemented. Measure actual transfer/synchronization costs.
- WebGPU/WASM fallback is configured, but actual operator placement and CPU
  fallback costs have not been measured. Do not assume all work runs on GPU.
- The original Python loop uses a single rotating prompt and backpropagation per
  update. Its backward cost differs, so the evaluation count is not evidence of
  a sevenfold runtime difference.

Prioritize **loss reduction per unit wall time**, as well as time per update.
Making noisy SPSA updates faster alone may still leave convergence impractical.

- [ ] Profile text encoding, each vector-estimator step, vocoding, WavLM,
  transfers, validation and model initialization separately after warm-up.
  Record objective-call counts, elapsed time and loss reduction per second.
- [ ] Compare full five-prompt validation at configurable intervals or only for
  promising candidates, rather than after every update. Keep a separate
  provisional training loss and fully validated best loss; never mark the target
  reached until the full, fixed validation objective is ≤ 0.24. The default
  unlimited run must continue until that confirmed threshold or user cancellation.
- [ ] Benchmark multi-direction averaging and structured/block search using
  equal wall-time and inference-call budgets; extra probes may improve each
  update but make the overall run slower.
- [ ] Evaluate staying near/restarting from the validated best style and adjusting
  step size, rather than letting unsuccessful updates accumulate unchecked.
- [ ] Investigate GPU-resident intermediates and buffer reuse, measuring actual
  savings while preserving deterministic inputs, cleanup and cancellation.
- [ ] Identify CPU fallback hot spots before attempting provider/operator changes;
  recheck numerical agreement and real-browser compatibility after changes.
- [ ] Keep duration-predictor runs sequential despite performance work: the
  previous overlapping asyncify calls stalled. Optimize this stage only with a
  verified runtime/session strategy and a regression check.

## Priority 3: robustness and performance

- [ ] Add a regression check for sequential duration-predictor calls and the
  previously stalled duration-stage end-to-end case.
- [ ] Measure per-model inference time, actual GPU/CPU operator placement,
  download size and peak memory on supported browsers/devices.
- [ ] Investigate safe GPU-resident intermediates / fewer CPU copies. Preserve
  explicit tensor cleanup and cancellation behavior.
- [ ] Compare browser speaking-rate measurement against Python librosa/scipy on
  the same reference and generated audio, including silence/short clips.
- [ ] Add model revision/hash information to the prepared manifest and reports.
- [ ] Make bounded experimental outputs clearly distinguish target reached
  from iteration limit in the main completion message, not just the report.

## Checks before a subsequent release

- [ ] Run `npm --prefix web test` and `npm --prefix web run build`.
- [ ] Prepare assets with `python tools/prepare_webgpu.py`, then run
  `python tests/verify_webgpu_assets.py`.
- [ ] Run real-browser extraction, cancellation, checkpoint/resume (once added),
  and output read-back checks.
- [ ] Publish measured convergence/quality results and limitations; keep SPSA
  labeled experimental until the evidence justifies stronger claims.
