# -*- coding: utf-8 -*-
"""Read config.yaml and hand out the settings the rest of the code uses.

Every tunable value lives in config.yaml beside this file. What stays here is
structural: the sample rates the released model works at, the shapes of the two
style vectors, and which device is available.
"""
import os

import torch
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR = 44100                 # SupertonicTTS output rate
WAVLM_SR = 16000           # what WavLM expects
TTL_SHAPE = (1, 50, 256)   # timbre style: 50 token rows, each on the unit sphere
DP_SHAPE = (1, 8, 16)      # duration style: 8 rows, likewise

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path=CONFIG_PATH):
    """Read a config file. Pass a path to run with settings other than the default."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()

ONNX_DIR = CFG["model"]["onnx_dir"]
PRESET_DIR = CFG["model"]["presets"]
TOTAL_STEP = CFG["model"]["total_step"]
SPEED = CFG["model"]["speed"]
SEED = CFG["model"]["seed"]
LAYER = CFG["timbre"]["layer"]
THRESHOLD = CFG["timbre"]["threshold"]
TEXTS = list(CFG["prompts"])
