"""Simulate a LoRA training run, writing trainer_state.json incrementally.

Used by the dashboard's "Launch training" button on machines without a GPU.
It produces the exact same HuggingFace ``trainer_state.json`` schema as the real
trainer and rewrites it after every logging step (with a short delay), so the
dashboard's live loss chart updates in real time.

The simulated curve mirrors real small-dataset behaviour: train loss decays
smoothly while eval loss bottoms out around epoch 2 and ticks back up.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

LOG_EVERY = 10
PEAK_LR = 2e-4


def lr_at(step: int, total: int, warmup: int, peak_lr: float) -> float:
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_loss_at(step: int, total: int, rng: random.Random) -> float:
    base = 0.32 + 1.55 * math.exp(-3.0 * step / total)
    return round(base + rng.uniform(-0.04, 0.04), 4)


def eval_loss_at(epoch: int, epochs: int, rng: random.Random) -> float:
    # U-shape: best near the second epoch, then mild overfit.
    mid = max(2, round(epochs * 0.66))
    val = 0.70 + 0.22 * (epoch - mid) ** 2 / max(epochs, 1)
    if epoch < mid:
        val = 0.70 + 0.25 * (mid - epoch)
    return round(val + rng.uniform(-0.01, 0.01), 4)


def grad_norm_at(step: int, total: int, rng: random.Random) -> float:
    base = 0.45 * math.exp(-2.0 * step / total) + 0.12
    return round(base + rng.uniform(-0.03, 0.03), 4)


def write_state(path: str, log_history: list[dict], epochs: int, total: int, step: int):
    evals = [r for r in log_history if "eval_loss" in r]
    best = min(evals, key=lambda r: r["eval_loss"]) if evals else None
    state = {
        "best_metric": best["eval_loss"] if best else None,
        "best_model_checkpoint": (
            f"{os.path.dirname(path)}/checkpoint-{best['step']}" if best else None
        ),
        "epoch": round(step / (total / epochs), 4),
        "global_step": step,
        "max_steps": total,
        "num_train_epochs": epochs,
        "log_history": log_history,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)  # atomic so the dashboard never reads a half-written file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=PEAK_LR)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument("--step-delay", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    state_path = os.path.join(args.output, "trainer_state.json")
    rng = random.Random(7)

    total = args.epochs * args.steps_per_epoch
    warmup = max(1, int(0.05 * total))
    log_history: list[dict] = []

    print(f"[simulate] mode=simulate epochs={args.epochs} rank={args.rank} "
          f"lr={args.learning_rate} steps={total}", flush=True)
    print(f"[simulate] writing {state_path} incrementally", flush=True)

    for step in range(LOG_EVERY, total + 1, LOG_EVERY):
        epoch = round(step / args.steps_per_epoch, 4)
        loss = train_loss_at(step, total, rng)
        log_history.append({
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": round(lr_at(step, total, warmup, args.learning_rate), 8),
            "grad_norm": grad_norm_at(step, total, rng),
        })
        print(f"[simulate] step {step}/{total}  epoch {epoch:.2f}  loss {loss:.4f}",
              flush=True)

        if step % args.steps_per_epoch == 0:
            ep = step // args.steps_per_epoch
            ev = eval_loss_at(ep, args.epochs, rng)
            log_history.append({
                "step": step,
                "epoch": float(ep),
                "eval_loss": ev,
                "eval_runtime": round(rng.uniform(8, 12), 3),
            })
            print(f"[simulate] *** eval @ epoch {ep}  eval_loss {ev:.4f}", flush=True)

        write_state(state_path, log_history, args.epochs, total, step)
        time.sleep(args.step_delay)

    best = min((r for r in log_history if "eval_loss" in r),
               key=lambda r: r["eval_loss"])
    print(f"[simulate] done. best eval_loss {best['eval_loss']} "
          f"at step {best['step']} (epoch {best['epoch']})", flush=True)


if __name__ == "__main__":
    main()
