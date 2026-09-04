"""Recover the timbre style by gradient descent on the WavLM objective.

Start at the nearest shipped preset, descend, and after every step put all 50
rows back on the unit sphere. Stop the moment the loss reaches the threshold.

Styles descend together: each reaches only its own output, so summing their
losses gives every vector the gradient it would get alone. A style that reaches
the threshold leaves the batch at once and a waiting one takes its place.
"""
import torch

from config import CFG
from style_file import preset_paths, project_to_sphere, read_style
from wavlm_loss import layer_stats, style_loss, to_wavlm_rate

C = CFG["timbre"]


def nearest_preset(targets, prompt, tts, wavlm):
    """For each target, (preset path, distance) of the closest preset under the objective."""
    reachable = {}
    for path in preset_paths():
        ttl, dp = read_style(path)
        wav = tts.synthesize(prompt, ttl, dp)
        with torch.no_grad():
            reachable[path] = layer_stats(wavlm, to_wavlm_rate(wav.reshape(1, -1)))
    return [min(((path, style_loss(stats, target).item())
                 for path, stats in reachable.items()), key=lambda pair: pair[1])
            for target in targets]


def invert(styles, prompts, tts, wavlm, on_reached=None):
    """Recover style_ttl for every entry, keeping the batch full as entries finish.

    Each entry carries "target" (reference statistics), "ttl" (starting vector)
    and "dp" (duration style to synthesize with). On reaching the threshold its
    "ttl" becomes the recovered vector, "loss" and "steps" are filled in, and
    on_reached(entry) is called.
    """
    waiting, active = list(styles), []

    def admit():
        while len(active) < C["batch"] and waiting:
            s = waiting.pop(0)
            s["ttl"] = s["ttl"].clone().requires_grad_(True)
            s["optimizer"] = torch.optim.Adam([s["ttl"]], lr=C["lr"])
            s["schedule"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                s["optimizer"], patience=C["lr_patience"], factor=C["lr_factor"],
                min_lr=C["lr"] * 0.01)
            s["loss"], s["best"] = float("inf"), None
            s["steps"], s["taken"] = 0, 0          # steps to the best loss, steps taken
            active.append(s)

    admit()
    step = 0
    while active:
        # one prompt per step for the whole batch; a global counter keeps the
        # rotation smooth when a finished style is replaced mid-run
        prompt = prompts[step % len(prompts)]
        step += 1
        ids, mask, latent, latent_mask = prompt.repeat(len(active))
        ttl = torch.cat([s["ttl"] for s in active], dim=0)
        dp = torch.cat([s["dp"] for s in active], dim=0)

        wav, _ = tts.forward(ids, mask, ttl, dp, latent, latent_mask)
        mean, std = layer_stats(wavlm, to_wavlm_rate(wav))
        losses = [style_loss((mean[i], std[i]), (s["target"][0][0], s["target"][1][0]))
                  for i, s in enumerate(active)]
        for s in active:
            s["optimizer"].zero_grad()
        torch.stack(losses).sum().backward()

        reached = []
        for s, loss in zip(active, losses, strict=True):
            value = loss.item()
            s["taken"] += 1
            if value < s["loss"]:
                s["loss"], s["steps"] = value, s["taken"]
                # the loss was measured on the projected vector from the previous
                # step, so keep a projected clone rather than the half-stepped one
                s["best"] = s["ttl"].detach().clone()
                with torch.no_grad():
                    project_to_sphere(s["best"])
            torch.nn.utils.clip_grad_norm_([s["ttl"]], C["grad_clip"])
            s["optimizer"].step()
            with torch.no_grad():
                project_to_sphere(s["ttl"].data)
            s["schedule"].step(value)
            if s["loss"] <= C["threshold"]:
                reached.append(s)

        for s in reached:
            active.remove(s)
            s["ttl"] = s.pop("best")
            for key in ("optimizer", "schedule", "taken"):
                del s[key]
            if on_reached:
                on_reached(s)
        if reached:
            admit()
    return styles
