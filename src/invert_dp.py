# -*- coding: utf-8 -*-
"""Recover the duration style: match the reference speaking rate.

WavLM cannot supply this signal. Its time-pooled objective discards exactly the
temporal information rate lives in, so rate is measured on audio and style_dp is
inverted through the frozen, differentiable duration predictor.

Everything is audio-only: no transcript of the reference, no text-derived
syllable counts, no speed grid, and no bounds on the answer. The measurement is
made in the model's own terms, by a fixed point on the model's actual output:

    rho starts at 1, the multiplier on the starting duration
    repeat:
        ask the duration predictor for the starting durations times rho, by
        inverting it onto those targets with the 8 rows kept on the sphere
        synthesize each prompt at the length that dp really commands
        measure the rate of those syntheses
        take a secant step in log space toward the reference rate
    until the rates match, or two moves in a row fail to improve the match, or
    the model stops turning length into tempo

Saturation is measured, not assumed. Once the commanded duration has moved far
enough and the rate has not followed, the acoustic model has stopped converting
length into tempo; the speaker is reported saturated at the best rho reached and
nothing is clipped beforehand.

The rate detector is identical on both sides, so its consistent undercount of
true syllables cancels in the ratio.
"""
import math

import librosa
import numpy as np
import torch
from scipy.signal import find_peaks

from config import CFG, DEVICE, SPEED, SR, WAVLM_SR


def nuclei_rate(wav16, min_distance_s=0.10, prominence=0.08):
    """Syllable nuclei per second: peaks of the smoothed RMS envelope over the
    trimmed clip. Language-agnostic, and applied identically to the reference
    and the synthesis so its bias cancels in the ratio. None if too short."""
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


def _fit_to_durations(prompts, targets, tts, dp_start, c):
    """Optimize the 8x16 unit-row style_dp so the predicted durations hit the
    targets. The prior pulls toward the starting vector, so among the many dp
    values giving the same duration the answer stays a plausible one."""
    dp = dp_start.clone().requires_grad_(True)
    anchor = dp_start.detach().clone()
    optimizer = torch.optim.Adam([dp], lr=c["invert_lr"])
    wanted = torch.tensor(targets, dtype=torch.float32, device=DEVICE)
    for _ in range(c["invert_steps"]):
        optimizer.zero_grad()
        predicted = torch.stack(
            [(tts.duration(p.ids, dp, p.mask) / SPEED).reshape(()) for p in prompts])
        loss = (((predicted - wanted) ** 2).mean()
                + c["invert_prior"] * ((dp - anchor) ** 2).sum())
        loss.backward()
        optimizer.step()
        with torch.no_grad():                    # 8 rows back onto the sphere
            dp /= dp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return dp.detach()


def recover(reference16, ttl, dp_start, prompts, tts, cfg=None):
    """Recover style_dp for one speaker. -> (style_dp, report)

    The report carries the reference and synthesized rates, the multiplier
    reached, why the iteration stopped, and the path it took, so any number can
    be traced without running it again.
    """
    c = (cfg or CFG)["duration"]
    reference_rate = nuclei_rate(reference16)
    if reference_rate is None:
        return dp_start, {"status": "reference too short"}

    start_durations = [tts.predict_seconds(p, dp_start) for p in prompts]
    start_rate = synthesized_rate(prompts, ttl, dp_start, tts)
    if start_rate is None:
        return dp_start, {"status": "starting synthesis unmeasurable"}

    wanted = math.log(reference_rate)
    log_rho, log_rate = [0.0], [math.log(start_rate)]
    slope = -1.0                                  # faster speech in a shorter clip
    best = {"error": abs(start_rate / reference_rate - 1), "rho": 1.0,
            "rate": start_rate, "dp": dp_start, "fit_error": 0.0}
    path = [[1.0, round(start_rate, 3)]]
    status, stalled = "out of iterations", 0

    for _ in range(c["max_iters"]):
        move = (wanted - log_rate[-1]) / slope
        move = max(-c["max_log_step"], min(c["max_log_step"], move))
        asked = math.exp(log_rho[-1] + move)
        targets = [d * asked for d in start_durations]

        dp = _fit_to_durations(prompts, targets, tts, dp_start, c)
        commanded = [tts.predict_seconds(p, dp) for p in prompts]
        rho = float(np.mean([a / b for a, b in zip(commanded, start_durations)]))
        rate = synthesized_rate(prompts, ttl, dp, tts)
        if rate is None:
            status = "unmeasurable"
            break

        x, y = math.log(rho), math.log(rate)
        path.append([round(rho, 3), round(rate, 3)])
        error = abs(rate / reference_rate - 1)
        if error < best["error"]:
            best = {"error": error, "rho": rho, "rate": rate, "dp": dp,
                    "fit_error": float(np.mean([abs(a / t - 1)
                                                for a, t in zip(commanded, targets)]))}
            stalled = 0
        else:
            stalled += 1

        if error < c["tolerance"]:
            status = "converged"
            break
        if stalled >= 2:
            # two moves in a row could not improve the match: what is left is
            # below what the detector resolves over these prompts
            status = "noise floor"
            break
        travelled = x - log_rho[0]
        if abs(travelled) >= c["saturation_move"] and \
                (y - log_rate[0]) / travelled > c["saturation_slope"]:
            # length moved a long way and tempo did not follow it
            status = "saturated"
            break
        # measure the model's own response, but only over a move big enough to
        # rise above the detector's noise; a flat local step keeps the old slope
        recent = x - log_rho[-1]
        if abs(recent) >= c["min_response_move"]:
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
