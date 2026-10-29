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
