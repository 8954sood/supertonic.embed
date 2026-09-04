"""Extract a complete voice style from every recording in a list.

One JSON per recording, with style_ttl (timbre, from WavLM layer 4) and style_dp
(duration, from the reference speaking rate) both filled in, in the format the
shipped presets use.

The list has one recording per line; "path|id" names the output, otherwise the
file's name is used. Blank lines and # comments are skipped.

    python src/run_extract_style_batch.py                     # src/wavlist.txt -> styles/
    python src/run_extract_style_batch.py my_list.txt --out styles

Timbre runs first for the whole list, several recordings at a time on the GPU.
Duration follows one at a time, because it changes the audio length. Settings
live in src/config.yaml.
"""
import argparse
import json
import os
import sys
import time

import librosa

from config import CFG, WAVLM_SR
from frozen_tts import FrozenTTS, build_prompts
from invert_dp import recover
from invert_ttl import invert, nearest_preset
from style_file import preset_name, read_style, write_style
from wavlm_loss import load_wavlm, target_from_file

DEFAULT_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wavlist.txt")


def read_list(path):
    """-> [{'id', 'wav'}] in file order; a bad line stops the run and names itself."""
    entries, seen = [], {}
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            wav, _, given = line.partition("|")
            wav, given = wav.strip(), given.strip()
            name = given or os.path.splitext(os.path.basename(wav))[0]
            if not os.path.exists(wav):
                raise SystemExit(f"{path}:{number}: no such file: {wav}")
            if name in seen:
                raise SystemExit(f"{path}:{number}: id {name!r} already used on line {seen[name]}")
            seen[name] = number
            entries.append({"id": name, "wav": wav})
    if not entries:
        raise SystemExit(f"{path}: no recordings listed")
    return entries


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Extract style_ttl and style_dp for every recording in a list.")
    ap.add_argument("wavlist", nargs="?", default=DEFAULT_LIST,
                    help="one recording per line, optionally 'path|id' (default: src/wavlist.txt)")
    ap.add_argument("--out", default="styles", help="directory for the style JSONs (default: styles)")
    args = ap.parse_args()

    entries = read_list(args.wavlist)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(entries)} recordings from {args.wavlist}\n"
          f"timbre: WavLM layer {CFG['timbre']['layer']}, stop at {CFG['timbre']['threshold']}, "
          f"{CFG['timbre']['batch']} at a time", flush=True)

    tts = FrozenTTS()
    wavlm = load_wavlm()
    prompts = build_prompts(tts)
    started = time.time()

    def elapsed():
        return f"({(time.time() - started) / 60:.1f} min)"

    # ---- where each recording starts from --------------------------------
    for e in entries:
        e["target"] = target_from_file(wavlm, e["wav"])
    chosen = nearest_preset([e["target"] for e in entries], prompts[0], tts, wavlm)
    for e, (path, distance) in zip(entries, chosen, strict=True):
        e["preset"] = preset_name(path)
        e["distance"] = round(distance, 4)
        e["ttl"], e["dp"] = read_style(path)
    print(f"presets chosen {elapsed()}", flush=True)

    # ---- timbre, several at a time ---------------------------------------
    finished = []

    def reached(e):
        finished.append(e)
        print(f"  timbre  [{len(finished):>4}/{len(entries)}] {e['id']:<20} {e['preset']}  "
              f"loss {e['loss']:.4f} at {e['steps']} steps {elapsed()}", flush=True)

    invert(entries, prompts, tts, wavlm, on_reached=reached)

    # ---- duration, one at a time, then the finished style -----------------
    report = []
    for n, e in enumerate(entries, 1):
        reference = librosa.load(e["wav"], sr=WAVLM_SR)[0]
        e["dp"], duration = recover(reference, e["ttl"], e["dp"], prompts, tts)
        print(f"  duration[{n:>4}/{len(entries)}] {e['id']:<20} "
              f"{duration['rate_start']:.2f} -> {duration['rate_final']:.2f} per second "
              f"against {duration['rate_ref']:.2f}, x{duration['rho']:.2f}, {duration['status']} "
              f"{elapsed()}", flush=True)
        style = os.path.join(args.out, f"{e['id']}.json")
        write_style(style, e["ttl"], e["dp"], source_file=e["wav"])
        report.append({"id": e["id"], "wav": e["wav"], "style": style,
                       "preset": e["preset"], "preset_distance": e["distance"],
                       "loss": round(e["loss"], 4), "steps": e["steps"], "duration": duration})

    report_path = os.path.join(args.out, "extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"config": CFG, "wavlist": args.wavlist, "styles": report}, f, indent=2)
    minutes = (time.time() - started) / 60
    print(f"\n{len(report)} styles -> {args.out}   {minutes:.1f} min, "
          f"{minutes / len(report):.2f} min per recording\nreport -> {report_path}")


if __name__ == "__main__":
    main()
