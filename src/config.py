"""Settings. Tunables come from config.yaml; only structural constants live here."""
import os

import torch
import yaml

def select_device(requested=None):
    requested = requested or os.environ.get("SUPERTONIC_DEVICE", "auto")
    if requested == "auto":
        # Keep the original CUDA/CPU default; MPS is an explicit native path.
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in ("cuda", "mps", "cpu"):
        raise ValueError("SUPERTONIC_DEVICE must be auto, cuda, mps or cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use an Apple Silicon Mac with an MPS-enabled PyTorch build")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(requested)


DEVICE = select_device()

SR = 44100                 # SupertonicTTS output rate
WAVLM_SR = 16000           # what WavLM expects
TTL_SHAPE = (1, 50, 256)   # timbre style: 50 token rows, each on the unit sphere
DP_SHAPE = (1, 8, 16)      # duration style: 8 rows, likewise

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
          encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ONNX_DIR = os.environ.get("SUPERTONIC_ONNX_DIR", CFG["model"]["onnx_dir"])
PRESET_DIR = os.environ.get("SUPERTONIC_PRESET_DIR", CFG["model"]["presets"])
TOTAL_STEP = CFG["model"]["total_step"]
SPEED = CFG["model"]["speed"]
SEED = CFG["model"]["seed"]
LAYER = CFG["timbre"]["layer"]
TEXTS = list(CFG["prompts"])
