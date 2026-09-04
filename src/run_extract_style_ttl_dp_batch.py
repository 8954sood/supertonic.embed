# -*- coding: utf-8 -*-
"""Extract a complete voice style from every recording in a list.

One command, one JSON per recording, with both halves filled in:

    style_ttl  [1,50,256]  the timbre, inverted against WavLM layer 4
    style_dp   [1,8,16]    the duration, matched to the reference speaking rate

The result has the same shape as the styles SupertonicTTS ships in
models/voice_styles, so it can be dropped in beside F1..F5 / M1..M5 and used
straight away. Nothing in the model is trained; only the style vector moves.

The list is one recording per line. A line may name an id after a pipe,
otherwise the file's own name is the id. Blank lines and # comments are skipped:

    examples/alice.wav
    examples/take_03.wav|bob

Usage:
    python src/run_extract_style_ttl_dp_batch.py                 # src/wavlist.txt
    python src/run_extract_style_ttl_dp_batch.py my_list.txt --out styles

Timbre runs first, for the whole list, several recordings descending together on
the GPU. Duration follows one recording at a time, because it sets the length of
the audio and speakers no longer share a tensor shape. Defaults: src/config.yaml.
"""
import argparse
import json
import os
import sys
import time

import librosa

from config import CFG, WAVLM_SR, load_config
from frozen_tts import FrozenTTS, build_prompts
from invert_dp import recover
from invert_ttl import invert, nearest_preset
from style_file import preset_name, read_style, write_style
from wavlm_loss import load_wavlm, target_from_file

DEFAULT_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wavlist.txt")


def read_list(path):
    """-> [{'id', 'wav'}] in file order, with bad lines named."""
    entries, seen = [], {}
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wav, _, given = line.partition("|")
            wav, given = wav.strip(), given.strip()
            name = given or os.path.splitext(os.path.basename(wav))[0]
            if not os.path.exists(wav):
                raise SystemExit(f"{path}:{number}: no such file: {wav}")
            if name in seen:
                raise SystemExit(f"{path}:{number}: id {name!r} already used on "
                                 f"line {seen[name]}")
            seen[name] = number
            entries.append({"id": name, "wav": wav})
    if not entries:
        raise SystemExit(f"{path}: no recordings listed")
    return entries


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="Extract style_ttl and style_dp for every recording in a list.")
    ap.add_argument("wavlist", nargs="?", default=DEFAULT_LIST,
                    help="one recording per line, optionally 'path|id' "
                         "(default: src/wavlist.txt)")
    ap.add_argument("--out", default="styles",
                    help="directory for the style JSONs (default: styles)")
    ap.add_argument("--config", default=None, help="use a config other than src/config.yaml")
    ap.add_argument("--timbre-only", action="store_true",
                    help="skip the duration stage and leave style_dp at the preset's")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else CFG
    entries = read_list(args.wavlist)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(entries)} recordings from {args.wavlist}\n"
          f"timbre: WavLM layer {cfg['timbre']['layer']}, stop at "
          f"{cfg['timbre']['threshold']}, {cfg['timbre']['batch']} at a time",
          flush=True)

    tts = FrozenTTS(cfg["model"]["onnx_dir"])
    wavlm = load_wavlm()
    prompts = build_prompts(tts, texts=cfg["prompts"], seed=cfg["model"]["seed"],
                            speed=cfg["model"]["speed"])
    started = time.time()

    # ---- where each recording starts from --------------------------------
    for e in entries:
        e["target"] = target_from_file(wavlm, e["wav"], cfg["timbre"]["layer"])
    for e, (path, distance) in zip(entries, nearest_preset(
            [e["target"] for e in entries], prompts[0], tts, wavlm,
            cfg["timbre"]["layer"])):
        e["preset"] = preset_name(path)
        e["distance"] = round(distance, 4)
        e["ttl"], e["dp"] = read_style(path)
    print(f"presets chosen ({(time.time() - started) / 60:.1f} min)", flush=True)

    # ---- timbre, several at a time ---------------------------------------
    done = []

    def reached(e):
        done.append(e)
        print(f"  timbre  [{len(done):>4}/{len(entries)}] {e['id']:<20} {e['preset']}  "
              f"loss {e['loss']:.4f} at {e['steps']} steps "
              f"({(time.time() - started) / 60:.1f} min)", flush=True)

    invert(entries, prompts, tts, wavlm, cfg, on_reached=reached)

    # ---- duration, one at a time, then write the finished style -----------
    report = []
    for n, e in enumerate(entries, 1):
        record = {"id": e["id"], "wav": e["wav"], "preset": e["preset"],
                  "preset_distance": e["distance"], "loss": round(e["loss"], 4),
                  "steps": e["steps"]}
        if args.timbre_only:
            record["duration"] = "skipped"
        else:
            reference = librosa.load(e["wav"], sr=WAVLM_SR)[0]
            e["dp"], record["duration"] = recover(reference, e["ttl"], e["dp"],
                                                  prompts, tts, cfg)
            d = record["duration"]
            print(f"  duration[{n:>4}/{len(entries)}] {e['id']:<20} "
                  f"{d['rate_start']:.2f} -> {d['rate_final']:.2f} per second "
                  f"against {d['rate_ref']:.2f}, x{d['rho']:.2f}, {d['status']} "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)
        record["style"] = os.path.join(args.out, f"{e['id']}.json")
        write_style(record["style"], e["ttl"], e["dp"], source_file=e["wav"])
        report.append(record)

    path = os.path.join(args.out, "extraction_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "wavlist": args.wavlist,
                   "timbre_only": args.timbre_only, "styles": report}, f, indent=2)
    minutes = (time.time() - started) / 60
    print(f"\n{len(report)} styles -> {args.out}")
    print(f"{minutes:.1f} min, {minutes / len(report):.2f} min per recording")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
