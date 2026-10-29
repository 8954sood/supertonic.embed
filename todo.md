# WebGPU extraction TODO

Last updated: 2026-10-30 (Asia/Seoul)

## Current state

- Browser-only extraction is implemented: reference audio → nearest preset →
  timbre optimization → duration recovery → style JSON, report and preview WAV.
- Inference uses ONNX Runtime Web 1.31.1 with WebGPU and WASM fallback.
- Timbre optimization now uses **preset-guided blockwise pattern search v1**.
  All ten starting presets are ranked on the full five-prompt average. Search
  uses tangent/geodesic trials toward presets, random row blocks and successful
  directions. Candidates must improve the full fixed average to be retained;
  rejected moves never accumulate. Adaptive radii replace timbre Adam moments.
- The default remains unlimited until a fully confirmed mean loss ≤ 0.24, with
  manual cancellation. Screening/audits and scale restarts do not stop the run.
  Attaining the threshold is not guaranteed. Duration fitting still uses SPSA.
- Best-style download is available during search and after cancellation, before
  duration recovery. It is marked partial and uses the initial duration preset.
  This is not a resumable optimizer checkpoint; save it before refresh/new run.
- Reports include initial/best and candidate loss, accepted moves, per-prompt
  best losses, evaluations, wall time, radius restarts and bounded loss history.
- Previous user-observed SPSA runs plateaued around 0.87 after hundreds of updates.
  A controlled single-reference comparison now runs through `/benchmark.html`:
  both algorithms start at M1 with identical noise/prompts and 300 prompt calls.
  See `docs/WEBGPU_VALIDATION.md` for measured results and limitations.
- Prepared model contracts and JavaScript tests pass. Per-model run/readback
  timers are implemented, but GPU operator placement and memory profiling remain
  pending. CPU-backed intermediates remain. Estimator invariant feeds are reused.
- Duration predictor calls remain sequential; overlapping asyncify runs previously
  stalled. This constraint also applies during benchmarking.

## Priority 1: preserve work and measure convergence

- [ ] Export the best style and optimizer state periodically, before target loss
  is reached; support downloading the current best without waiting for duration
  recovery. Best-style download is implemented; optimizer-state export is pending.
- [ ] Support resuming checkpoints, including current/best style, search radii/directions,
  iteration, RNG state, prompt/noise settings, target objective and model version.
- [ ] Record initial, current and best loss, accepted improvements, objective-call
  count, elapsed time and per-prompt loss. Show a loss history and plateau status.
  A plateau notification must not silently stop the requested unlimited run.
- [ ] Build a repeatable, consented/CC0 reference-audio benchmark using fixed
  model revisions and seeds. Compare equal wall time and equal objective-call
  budgets, not iteration counts alone.

## Priority 2: improve timbre optimization

The following include legacy SPSA experiments and further search hypotheses.
Implemented search behavior is documented above; broader evaluation is pending.

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
- [x] Rank all starting presets on the fixed five-prompt validation objective.
- [ ] Compare multiple search starts and preset mixtures across references.
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

- Current search uses 2 prompt evaluations per screened-out trial, or 6 when
  fully validating the selected candidate (reuse one probe + four more prompts).
  Initialization costs 5 and full preset ranking costs 50. These are not comparable
  iteration units to the former SPSA loop's average 3 evaluations/update.
- Each prompt evaluation still uses 8 model runs: encoder, 5 estimator steps,
  vocoder and WavLM. Cost varies with candidate acceptance and screening.
- Each session output is read with `getData()` and copied into a Float32Array,
  then provided as CPU-backed input to the next session. GPU-resident chaining
  has not been implemented. Measure actual transfer/synchronization costs.
- WebGPU/WASM fallback is configured, but actual operator placement and CPU
  fallback costs have not been measured. Do not assume all work runs on GPU.
- The original Python loop uses a single rotating prompt and backpropagation per
  update. Its backward cost differs, so the evaluation count is not evidence of
  a sevenfold runtime difference.

Prioritize **loss reduction per unit wall time**, as well as time per update.
A faster trial alone does not establish faster convergence or voice quality.

- [ ] Profile text encoding, each vector-estimator step, vocoding, WavLM,
  transfers, validation and model initialization separately after warm-up.
  Record objective-call counts, elapsed time and loss reduction per second.
- Timers and objective/loss telemetry are implemented; collect real-model samples
  and compare warmed timings before marking profiling complete.
- [ ] Compare full five-prompt validation at configurable intervals or only for
  promising candidates, rather than after every update. Keep a separate
  provisional training loss and fully validated best loss; never mark the target
  reached until the full, fixed validation objective is ≤ 0.24. The default
  unlimited run must continue until that confirmed threshold or user cancellation.
- Candidate screening plus periodic full audits are implemented. A single CC0
  equal-call comparison against legacy interval-5 SPSA is recorded; extend it
  across references/seeds and held-out text before claiming general superiority.
- [ ] Benchmark multi-direction averaging and structured/block search using
  equal wall-time and inference-call budgets; extra probes may improve each
  update but make the overall run slower.
- [ ] Evaluate staying near/restarting from the validated best style and adjusting
  step size. Implemented via strict acceptance and adaptive radii; broader
  comparison remains pending.
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
- [x] Make bounded experimental outputs clearly distinguish target reached
  from iteration limit in the main completion message, not just the report.

## Checks before a subsequent release

- [ ] Run `npm --prefix web test` and `npm --prefix web run build`.
- [ ] Prepare assets with `python tools/prepare_webgpu.py`, then run
  `python tests/verify_webgpu_assets.py`.
- [ ] Run real-browser extraction, cancellation, checkpoint/resume (once added),
  and output read-back checks.
- [ ] Publish measured convergence/quality results and limitations; keep browser search
  labeled experimental until the evidence justifies stronger claims.
