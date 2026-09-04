"""The voice-style file: read, write, presets, and the unit sphere.

A style is two tensors in one JSON, in the format SupertonicTTS ships:
style_ttl [1,50,256] is the timbre, style_dp [1,8,16] the duration. Every row
of both is a unit vector in the released presets, so an extracted style is
projected back to unit rows after every optimizer step.
"""
import glob
import json
import os
from datetime import datetime

import torch

from config import DEVICE, DP_SHAPE, PRESET_DIR, SR, TTL_SHAPE


def read_style(path):
    """-> (style_ttl, style_dp) on the device."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return tuple(torch.tensor(d[k]["data"], dtype=torch.float32,
                              device=DEVICE).reshape(*d[k]["dims"])
                 for k in ("style_ttl", "style_dp"))


def write_style(path, style_ttl, style_dp, source_file=None):
    """Write a style the way SupertonicTTS ships them, so it drops in beside the presets."""
    payload = {
        "style_ttl": {"data": style_ttl.detach().cpu().numpy().tolist(),
                      "dims": list(TTL_SHAPE), "type": "float32"},
        "style_dp": {"data": style_dp.detach().cpu().numpy().tolist(),
                     "dims": list(DP_SHAPE), "type": "float32"},
        "metadata": {"source_file": source_file or "unknown",
                     "source_sample_rate": SR, "target_sample_rate": SR,
                     "extracted_at": datetime.now().isoformat()},
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def preset_paths():
    """The shipped presets, F1-F5 and M1-M5, in name order."""
    return sorted(glob.glob(os.path.join(PRESET_DIR, "[FM]*.json")))


def preset_path(name):
    """'M1' -> 'models/voice_styles/M1.json'."""
    return os.path.join(PRESET_DIR, f"{name}.json")


def preset_name(path):
    """'models/voice_styles/M1.json' -> 'M1'."""
    return os.path.splitext(os.path.basename(path))[0]


def project_to_sphere(style):
    """Put every row back at unit L2 norm, in place."""
    style /= style.norm(dim=-1, keepdim=True).clamp_min(1e-8)
