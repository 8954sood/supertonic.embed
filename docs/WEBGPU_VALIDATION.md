# WebGPU implementation validation — 2026-10-29

Environment: local macOS ARM64, in-app Chromium browser, ONNX Runtime Web 1.31.1.
Preparation used Python 3.12, PyTorch 2.15.0, Transformers 4.57.6 and ONNX Runtime
1.31.1. Model inputs were the released Supertonic-2 and microsoft/wavlm-large.

Checks completed:

- WavLM first-four-block statistics match the original full model's
  `hidden_states[4]` before its final encoder normalization.
- ONNX/PyTorch statistics agree at 32,000 and 48,000 audio samples (rtol 3e-3,
  atol 3e-4).
- All five ONNX graphs pass the ONNX checker and match browser input contracts.
- Nine JavaScript tests pass, including unlimited optimization stopping at the
  target threshold and cancellation.
- Production static build succeeds, with WASM explicitly bundled by URL.
- Browser end-to-end run on the repository's CC0 `examples/01.wav` completed
  preset selection, timbre optimization, duration recovery, JSON and WAV output.

The end-to-end run was deliberately bounded to one timbre update and one update
per duration fit, to exercise the entire path. It selected M1, returned timbre
loss 0.9261751981385158 (`iteration limit`), and stopped duration recovery with
`no improvement` after two outer iterations. These results do not demonstrate
convergence or voice-cloning quality. Default application settings remain
unlimited timbre updates until loss ≤ 0.24 and 100 updates per duration fit.

Downloaded output files were independently read back: both style tensors have
the required dimensions and finite unit rows (maximum norm error below 3e-8).
The preview is a valid 5.433-second, mono, 16-bit PCM, 44.1 kHz WAV.

A browser stall was reproduced at duration recovery in the initial version.
Replacing overlapping duration-predictor session runs with sequential runs
allowed the same bounded end-to-end case to finish. Do not parallelize calls
into this asyncify ORT session. Extraction now runs in a dedicated worker;
its cancellation was also exercised in the browser. Duration progress logs
include each synthesized sentence and optimizer step counts.

## Performance changes — subsequent checks, 2026-10-29

Historical interval-five SPSA changes; the algorithm was subsequently replaced
by the 2026-10-30 search described below.

The current default validates every five timbre updates, at initialization and
at the final bounded update. Interval 1 retains the previous schedule. Both
settings accept best styles and stop only on the full fixed-prompt objective.
Unvalidated steps expose probe losses separately. Unlimited mode and the 0.24
threshold remain in effect. This schedule may miss intermediate improvements;
its effect on real-audio convergence has not been established.

Validation on the current checkout:

- All 16 JavaScript tests pass. A deterministic synthetic objective exercises
  both validation schedules: 100 updates require 705 vs 305 prompt evaluations,
  including initialization, excluding preset selection. Both return a measured
  best. This counts objective work, not real GPU execution or a voice benchmark.
- Regression cases cover probe losses below the target without false success,
  final bounded validation, retention of the measured best after regression,
  cancellation between probes and tensor cleanup on inference/readback failure.
- Session-runner tests verify that borrowed input tensors survive repeated runs
  and that all pending output reads finish before outputs are disposed on error.
- Production build passes.

Per-session load, run and readback/copy timers and per-vector-step timers are
available in reports. The first call is separated from subsequent calls, but
dynamic shapes can cause compilation later too. Timers measure overlapping wall
time; they do not identify operator placement or isolate GPU transfer time.

This checkout contains no prepared model assets. Asset preparation/ONNX checks
and a new real-browser extraction have **not** been run for these changes. The
earlier browser result above validates the earlier version only. Real-device
wall-time savings, convergence, memory use and voice quality still need a fixed
reference/seed comparison at intervals 1 and 5. GPU-resident chaining and actual
provider-placement profiling remain pending. Duration predictor calls remain
sequential.

## Algorithm replacement — 2026-10-30

Timbre SPSA/Adam is replaced by preset-guided blockwise pattern search v1.
Released presets provide structured directions; random row blocks allow residuals
outside their span. Trials move on row spheres, screen plus/minus on a rotating
prompt, reuse that measurement for full validation and accept only an improvement
in the fixed five-prompt mean. Accepted directions receive follow-up trials.
Failures shrink the radius; sustained failure restarts a radius ladder around the
best. No unsuccessful position or Adam state accumulates. Default unlimited mode
and the fully validated 0.24 stopping rule are preserved. The duration algorithm
and sequential session calls are unchanged.

The application now selects its initial preset using all five prompts (50 prompt
evaluations). It also makes the current best style downloadable during the run
and after cancellation, marked partial with duration recovery incomplete. This is
not optimizer resume support.

### Controlled browser comparison

Local macOS ARM64, in-app Chromium, ORT Web 1.31.1; repository CC0 `examples/01.wav`,
seed 42, five sampling steps and fixed five-prompt noise. Both methods start at
M1 with M1 duration style, deliberately excluding the new preset-selection change.
Each receives exactly **300 TTS + WavLM prompt evaluations**, including optimizer
initialization, excluding loading, preset selection and duration recovery. Legacy
SPSA uses validation interval 5 and its existing learning/perturbation settings.
Only completed full validations may change the reported best; partial evaluations
at budget exhaustion count toward time and calls but cannot accept a style.

First run (legacy then pattern):

| Algorithm | Initial mean | Best mean | Prompt evaluations | Optimization wall time |
| --- | ---: | ---: | ---: | ---: |
| Legacy SPSA/Adam | 0.92617520 | 0.91197908 | 300 | 46.334 s |
| Pattern search v1 | 0.92617520 | 0.71512739 | 300 | 46.228 s |

Reversed-order run reproduced both best losses exactly: pattern first took
46.950 s and legacy second 46.821 s, again 300 prompt evaluations each.
At 45 seconds in that run, pattern reached 0.71512739 and legacy 0.91333915.

At a common **45-second** cutoff, the recorded fully validated bests are likewise
0.91197908 and 0.71512739. This is a single-reference/seed comparison, not evidence
of general voice quality, CUDA parity, or eventual attainment of 0.24. It supports
the new search on this bounded case only. Speaker similarity, intelligibility,
held-out prompts and broader seed/reference comparisons remain pending.

Reproduce using the `/benchmark.html` development page described in
`web/README.md`. The page records per-trial loss, prompt calls and wall times and
supports reversing execution order to inspect cache/order effects.

The existing prepared assets pass `tests/verify_webgpu_assets.py`; assets were not
regenerated during this algorithm change. Exact SHA-256 identifiers:

```text
examples/01.wav       57f25abc75c2e7cc4d3c9a6e45f54160034bc714b2ba952407dae7f91d6d31f3
manifest.json         e94c864100cda34e9c7d75fa7726f9db1466c82727c12ed6140abad054e569bb
duration_predictor    6d556b3691165c364be91dc0bd894656b5949f5acd2750d8ec2f954010845011
text_encoder          dd5f535ed629f7df86071043e15f541ce1b2ab7f1bdbce4c7892b307bca79fa3
vector_estimator      105e9d66fd8756876b210a6b4aa03fc393b1eaca3a8dadcc8d9a3bc785c86a35
vocoder               19bd51f47a186069c752403518a40f7ea4c647455056d2511f7249691ecddf7c
wavlm_stats           891730057b14d0077a1696fa6301ffb81da9dd94cf00998ba92fb0c63e17e126
```

All 22 JavaScript tests and the production build pass. New tests cover monotone
full-objective acceptance, rejection of misleading single-prompt gains, every
style row remaining reachable, radius restarts without best-style drift, sphere
geometry, reproducibility, cancellation, non-finite loss and identical-guide
fallback. The legacy optimizer remains available for duration fitting and the
controlled comparison.

### Browser integration and file read-back

The new application completed extraction on `examples/01.wav` with one timbre
trial and one update per duration fit. Full-objective preset ranking selected M1;
the new trial reduced mean loss from 0.92617520 to 0.91465325. It completed duration
recovery (`no improvement`) and clearly displayed `iteration limit / target not
reached` rather than success. The partial best-style link was also present.

Downloaded partial/final JSON files were read independently. Both style tensors
have the expected dimensions, finite values and unit row norms within 1e-6.
The partial file carries `partial: true`, `duration_recovered: false` and the new
algorithm ID. The final file carries the iteration-limit status and the same
fully validated loss. Preview read-back confirms a 5.433-second mono PCM16
44.1 kHz WAV. This bounded smoke case verifies plumbing, not target convergence.

A separate unlimited run was cancelled after reaching mean loss 0.80571448. The
worker stopped, Start became enabled, and the best-style download remained visible
and usable. Its downloaded JSON was independently read and retained the validated
loss, correct timbre size, finite values and partial/duration-incomplete markers.
