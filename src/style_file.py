# -*- coding: utf-8 -*-
"""The voice-style file: read one, write one, and keep it on the manifold.

A style is two tensors in one JSON, in the format SupertonicTTS ships:

    style_ttl  [1,50,256]  the timbre, the only part that reaches the audio
    style_dp   [1,8,16]    the duration, which sets how long an utterance runs

Every row of both is a unit vector in the released presets, to float precision.
That is the surface the withheld encoder writes onto, so an extracted style is
projected back to it after every optimizer step.
"""
import glob
import json
import os
from datetime import datetime

import torch

from config import DEVICE, DP_SHAPE, PRESET_DIR, SR, TTL_SHAPE


def read_style(path):
    """A style JSON as (style_ttl, style_dp), on the configured device."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for key in ("style_ttl", "style_dp"):
        part = d[key]
        out.append(torch.tensor(part["data"], dtype=torch.float32,
                                device=DEVICE).reshape(*part["dims"]))
    return tuple(out)


def write_style(path, style_ttl, style_dp, source_file=None):
    """Write a style in the format SupertonicTTS itself consumes, so the result
    can be dropped in beside the shipped presets and used as one."""
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


def preset_paths(preset_dir=None):
    """The shipped presets, F1-F5 and M1-M5, in name order."""
    return sorted(glob.glob(os.path.join(preset_dir or PRESET_DIR, "[FM]*.json")))


def preset_path(name, preset_dir=None):
    """The path of one preset, given its name (for example 'M1')."""
    return os.path.join(preset_dir or PRESET_DIR, f"{name}.json")


def preset_name(path):
    """'models/voice_styles/M1.json' -> 'M1'."""
    return os.path.splitext(os.path.basename(path))[0]


def project_to_sphere(style):
    """Put every token row back at unit L2 norm, in place."""
    style /= style.norm(dim=-1, keepdim=True).clamp_min(1e-8)
