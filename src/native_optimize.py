"""Projected Adam with a fixed validation objective and periodic best recovery."""
import math
import time

import torch


def optimize_style(initial, objective, prompt_count, *, iterations=0, threshold=0.24,
                   validation_interval=5, rate=2e-4, recovery_patience=20,
                   on_step=lambda event: None, on_best=lambda value, event: None):
    if iterations < 0 or validation_interval < 1 or prompt_count < 1 or recovery_patience < 1:
        raise ValueError("Invalid optimizer settings")
    if not math.isfinite(threshold) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid threshold or learning rate")
    x = initial.detach().clone()
    if not torch.isfinite(x).all() or torch.any(x.norm(dim=-1) < 1e-8):
        raise ValueError("Invalid initial style")
    x /= x.norm(dim=-1, keepdim=True)
    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=rate)
    best = x.detach().clone()
    steps = evaluations = backward_calls = improvements = recoveries = last_improvement = 0
    failed_validations = 0
    started = time.perf_counter()

    def evaluate(index):
        nonlocal evaluations
        loss = objective(x, index)
        evaluations += 1
        if loss.numel() != 1 or not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite or non-scalar objective")
        return loss

    def validate():
        with torch.no_grad():
            return [evaluate(i).item() for i in range(prompt_count)]

    best_prompt_losses = validate()
    best_loss = initial_loss = sum(best_prompt_losses) / prompt_count

    def event(**extra):
        return {"step": steps, "initial_loss": initial_loss, "best_loss": best_loss,
                "best_prompt_losses": list(best_prompt_losses), "evaluations": evaluations,
                "backward_calls": backward_calls, "improvements": improvements,
                "recoveries": recoveries, "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
                "steps_since_improvement": steps - last_improvement, **extra}

    on_best(best, event())
    on_step(event(validated=True, candidate_loss=best_loss, accepted=True))
    interrupted = False
    try:
        while best_loss > threshold and (iterations == 0 or steps < iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = evaluate(steps % prompt_count)
            training_loss = loss.item()
            loss.backward()
            backward_calls += 1
            if x.grad is None or not torch.isfinite(x.grad).all().item():
                raise RuntimeError("Missing or non-finite style gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_([x], 1.0).item()
            optimizer.step()
            with torch.no_grad():
                x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            steps += 1
            validated = steps % validation_interval == 0 or steps == iterations
            candidate_loss = None
            accepted = recovered = False
            if validated:
                losses = validate()
                candidate_loss = sum(losses) / prompt_count
                if candidate_loss < best_loss:
                    best_loss, best_prompt_losses = candidate_loss, losses
                    best = x.detach().clone()
                    improvements += 1; last_improvement = steps; failed_validations = 0
                    accepted = True
                    on_best(best, event())
                else:
                    failed_validations += 1
                    if failed_validations >= recovery_patience:
                        with torch.no_grad():
                            x.copy_(best)
                        optimizer.state.clear()
                        optimizer.param_groups[0]["lr"] = max(rate * 0.01, optimizer.param_groups[0]["lr"] * 0.5)
                        failed_validations = 0; recoveries += 1; recovered = True
            on_step(event(training_loss=training_loss, gradient_norm=grad_norm, validated=validated,
                          candidate_loss=candidate_loss, accepted=accepted, recovered=recovered))
    except KeyboardInterrupt:
        interrupted = True
    return best, event(status="cancelled" if interrupted else "threshold reached" if best_loss <= threshold else "iteration limit")
