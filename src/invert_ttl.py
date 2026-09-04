# -*- coding: utf-8 -*-
"""Recover the timbre style: gradient descent on the WavLM objective.

The vector starts at the nearest shipped preset and descends until the loss
reaches the threshold. After every optimizer step all 50 token rows are put back
on the unit sphere, the surface the withheld encoder writes onto; without that
the vector drifts off the manifold and the synthesis degrades.

Reaching the threshold is the only stopping condition. It is not a convergence
point: synthesizing two different sentences from the same preset leaves a
residual caused by content rather than by speaker, and 0.24 sits inside the band
that residual spans, so a run that reaches it is already at same-voice distance.

Styles are recovered together. Each one reaches only its own output, so stacking
B of them into one forward pass and SUMMING their losses gives every vector the
gradient it would have received alone; averaging would divide each speaker's
learning rate by B. Two rules keep the batch honest: a style that reaches the
threshold leaves it at once, because further descent walks the vector into
territory the frozen model never saw, and a waiting style takes its place so the
batch stays full.
"""
import torch

from config import CFG
from style_file import preset_paths, project_to_sphere, read_style
from wavlm_loss import layer_stats, style_loss, to_wavlm_rate


def nearest_preset(targets, prompt, tts, wavlm, layer):
    """For each target, the preset whose synthesis is closest under the objective.

    Returns [(path, distance)]. The distance is the loss the descent starts from.
    """
    reachable = {}
    for path in preset_paths():
        ttl, dp = read_style(path)
        wav = tts.synthesize(prompt, ttl, dp)
        with torch.no_grad():
            reachable[path] = layer_stats(wavlm, to_wavlm_rate(wav.reshape(1, -1)), layer)
    return [min(((path, style_loss(stats, target).item())
                 for path, stats in reachable.items()), key=lambda pair: pair[1])
            for target in targets]


def invert(styles, prompts, tts, wavlm, cfg=None, on_reached=None):
    """Recover style_ttl for every entry, keeping the batch full as they finish.

    Each entry is a dict carrying "target" (the reference statistics), "ttl"
    (the starting vector) and "dp" (the duration style to synthesize with). When
    an entry reaches the threshold its "ttl" is replaced by the recovered vector
    and "loss" and "steps" are filled in, then `on_reached(entry)` is called.
    """
    c = (cfg or CFG)["timbre"]
    layer, threshold, batch = c["layer"], c["threshold"], c["batch"]
    waiting, active = list(styles), []

    def admit():
        while len(active) < batch and waiting:
            s = waiting.pop(0)
            s["ttl"] = s["ttl"].clone().requires_grad_(True)
            s["optimizer"] = torch.optim.Adam([s["ttl"]], lr=c["lr"])
            s["schedule"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                s["optimizer"], patience=c["lr_patience"], factor=c["lr_factor"],
                min_lr=c["lr"] * 0.01)
            s["loss"], s["steps"], s["best"], s["age"] = float("inf"), 0, None, 0
            active.append(s)

    admit()
    step = 0
    while active:
        # one prompt per step for the whole batch, so the batch stays
        # rectangular; a global counter keeps the rotation smooth when a
        # finished style is replaced mid-run
        prompt = prompts[step % len(prompts)]
        step += 1
        ids, mask, latent, latent_mask = prompt.repeat(len(active))
        ttl = torch.cat([s["ttl"] for s in active], dim=0)
        dp = torch.cat([s["dp"] for s in active], dim=0)

        wav, _ = tts.forward(ids, mask, ttl, dp, latent, latent_mask)
        mean, std = layer_stats(wavlm, to_wavlm_rate(wav), layer)
        losses = [style_loss((mean[i], std[i]),
                             (s["target"][0][0], s["target"][1][0]))
                  for i, s in enumerate(active)]
        for s in active:
            s["optimizer"].zero_grad()
        torch.stack(losses).sum().backward()

        reached = []
        for s, loss in zip(active, losses):
            value = loss.item()
            s["age"] += 1
            if value < s["loss"]:
                s["loss"], s["steps"] = value, s["age"]
                # the loss was measured on the already-projected vector from the
                # previous step, and that is what will be synthesized, so keep a
                # projected clone rather than the half-stepped live one
                s["best"] = s["ttl"].detach().clone()
                with torch.no_grad():
                    project_to_sphere(s["best"])
            torch.nn.utils.clip_grad_norm_([s["ttl"]], c["grad_clip"])
            s["optimizer"].step()
            with torch.no_grad():
                project_to_sphere(s["ttl"].data)
            s["schedule"].step(value)
            if s["loss"] <= threshold:
                reached.append(s)

        for s in reached:
            active.remove(s)
            s["ttl"] = s.pop("best")
            for key in ("optimizer", "schedule", "age"):
                del s[key]
            if on_reached:
                on_reached(s)
        if reached:
            admit()
    return styles
