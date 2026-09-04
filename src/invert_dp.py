"""Recover the duration style by matching the reference speaking rate.

A time-pooled loss cannot see tempo, so rate is measured on audio and style_dp
is inverted through the frozen duration predictor. A fixed-point iteration on a
duration multiplier rho does it: fit dp to the starting durations times rho,
synthesize at the length that dp commands, measure the rate, take a secant step
in log space toward the reference rate. It stops when the rates match, when two
moves in a row fail to improve, or when length stops turning into tempo.

The same detector runs on reference and synthesis, so its bias cancels.
"""
import math

import librosa
import numpy as np
import torch
from scipy.signal import find_peaks

from config import CFG, DEVICE, SPEED, SR, WAVLM_SR

C = CFG["duration"]


def nuclei_rate(wav16, min_distance_s=0.10, prominence=0.08):
    """Syllable nuclei per second: peaks of the smoothed RMS envelope of the
    trimmed clip. None if the clip is too short to measure."""
    trimmed, _ = librosa.effects.trim(wav16, top_db=35)
    seconds = len(trimmed) / WAVLM_SR
    if seconds < 0.3:
        return None
    envelope = librosa.feature.rms(y=trimmed, frame_length=400, hop_length=160)[0]
    envelope = np.convolve(envelope, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(envelope,
                          distance=max(1, int(min_distance_s * WAVLM_SR / 160)),
                          prominence=prominence * envelope.max())
    return len(peaks) / seconds


def synthesized_rate(prompts, ttl, dp, tts):
    """Mean rate over the prompts, each synthesized at the length dp commands."""
    rates = []
    for prompt in prompts:
        wav, _ = tts.synthesize_at_own_length(prompt, ttl, dp)
        rate = nuclei_rate(librosa.resample(wav, orig_sr=SR, target_sr=WAVLM_SR))
        if rate is not None:
            rates.append(rate)
    return float(np.mean(rates)) if rates else None


def _fit_to_durations(prompts, targets, tts, dp_start):
    """The unit-row style_dp whose predicted durations hit the targets. A prior
    toward the starting vector keeps the answer plausible among the many dp
    values that give the same duration."""
    dp = dp_start.clone().requires_grad_(True)
    anchor = dp_start.detach().clone()
    optimizer = torch.optim.Adam([dp], lr=C["invert_lr"])
    wanted = torch.tensor(targets, dtype=torch.float32, device=DEVICE)
    for _ in range(C["invert_steps"]):
        optimizer.zero_grad()
        predicted = torch.stack(
            [(tts.duration(p.ids, dp, p.mask) / SPEED).reshape(()) for p in prompts])
        loss = ((predicted - wanted) ** 2).mean() + C["invert_prior"] * ((dp - anchor) ** 2).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dp /= dp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return dp.detach()


def recover(reference16, ttl, dp_start, prompts, tts):
    """Recover style_dp for one speaker. -> (style_dp, report)

    The report holds the rates, the multiplier reached, why the iteration stopped
    and the path it took, so every number can be traced without rerunning.
    """
    reference_rate = nuclei_rate(reference16)
    if reference_rate is None:
        return dp_start, {"status": "reference too short"}
    start_durations = [tts.predict_seconds(p, dp_start) for p in prompts]
    start_rate = synthesized_rate(prompts, ttl, dp_start, tts)
    if start_rate is None:
        return dp_start, {"status": "starting synthesis unmeasurable"}

    wanted = math.log(reference_rate)
    log_rho, log_rate = [0.0], [math.log(start_rate)]
    slope = -1.0                                  # shorter clip, faster speech
    best = {"error": abs(start_rate / reference_rate - 1), "rho": 1.0,
            "rate": start_rate, "dp": dp_start, "fit_error": 0.0}
    path = [[1.0, round(start_rate, 3)]]
    status, stalled = "out of iterations", 0

    for _ in range(C["max_iters"]):
        move = (wanted - log_rate[-1]) / slope
        move = max(-C["max_log_step"], min(C["max_log_step"], move))
        asked = math.exp(log_rho[-1] + move)
        targets = [d * asked for d in start_durations]

        dp = _fit_to_durations(prompts, targets, tts, dp_start)
        commanded = [tts.predict_seconds(p, dp) for p in prompts]
        rho = float(np.mean([a / b for a, b in zip(commanded, start_durations, strict=True)]))
        rate = synthesized_rate(prompts, ttl, dp, tts)
        if rate is None:
            status = "unmeasurable"
            break

        x, y = math.log(rho), math.log(rate)
        path.append([round(rho, 3), round(rate, 3)])
        error = abs(rate / reference_rate - 1)
        if error < best["error"]:
            fit_error = float(np.mean([abs(a / t - 1) for a, t in zip(commanded, targets, strict=True)]))
            best = {"error": error, "rho": rho, "rate": rate, "dp": dp, "fit_error": fit_error}
            stalled = 0
        else:
            stalled += 1

        if error < C["tolerance"]:
            status = "converged"
            break
        if stalled >= 2:                          # below what the detector resolves
            status = "noise floor"
            break
        travelled = x - log_rho[0]
        if abs(travelled) >= C["saturation_move"] and \
                (y - log_rate[0]) / travelled > C["saturation_slope"]:
            status = "saturated"                  # length moved, tempo did not follow
            break
        recent = x - log_rho[-1]                  # update the slope only over a clear move
        if abs(recent) >= C["min_response_move"]:
            measured = (y - log_rate[-1]) / recent
            if measured < -0.05:
                slope = max(-3.0, min(-0.33, measured))
        log_rho.append(x)
        log_rate.append(y)

    row_norms = best["dp"].norm(dim=-1).flatten().tolist()
    report = {"rate_ref": round(reference_rate, 3),
              "rate_start": round(start_rate, 3),
              "rate_final": round(best["rate"], 3),
              "rho": round(best["rho"], 3),
              "status": status,
              "iters": len(path) - 1,
              "path": path,
              "duration_fit_error": round(best["fit_error"], 4),
              "seconds_start": round(start_durations[0], 2),
              "seconds_final": round(start_durations[0] * best["rho"], 2),
              "row_norm_error": round(max(abs(n - 1) for n in row_norms), 7)}
    return best["dp"], report
