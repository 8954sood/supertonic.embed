"""Settings. Tunables come from config.yaml; only structural constants live here."""
import os

import torch
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR = 44100                 # SupertonicTTS output rate
WAVLM_SR = 16000           # what WavLM expects
TTL_SHAPE = (1, 50, 256)   # timbre style: 50 token rows, each on the unit sphere
DP_SHAPE = (1, 8, 16)      # duration style: 8 rows, likewise

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
          encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ONNX_DIR = CFG["model"]["onnx_dir"]
PRESET_DIR = CFG["model"]["presets"]
TOTAL_STEP = CFG["model"]["total_step"]
SPEED = CFG["model"]["speed"]
SEED = CFG["model"]["seed"]
LAYER = CFG["timbre"]["layer"]
TEXTS = list(CFG["prompts"])
