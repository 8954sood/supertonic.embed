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
- User-observed plateau: **478 timbre iterations, best loss 0.88410**. This is an
  observation from the running UI, not a controlled convergence benchmark. No
  optimizer improvements addressing this plateau have been applied yet.

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
