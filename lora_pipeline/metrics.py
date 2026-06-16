"""Training-metrics analysis engine.

Parses a HuggingFace ``Trainer`` ``trainer_state.json`` (or the ``log_history``
list it contains) and turns the raw logs into the loss / overfit / gradient
diagnostics described in Phase 3 of the training plan.

The thresholds encoded here come straight from the plan's
"What to watch during training" table:

    | Metric                  | Healthy             | Warning                         |
    |-------------------------|---------------------|---------------------------------|
    | Train loss              | Decreasing steadily | Spikes or flat after epoch 1    |
    | Eval loss               | Decreasing w/ train | Rising while train falls        |
    | Loss gap (train - eval) | < 0.3               | > 0.5 = overfit                 |
    | Gradient norm           | Stable ~0.1-0.5     | Exploding (> 2.0) = reduce LR   |
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

# Thresholds taken directly from the plan.
LOSS_GAP_HEALTHY = 0.3
LOSS_GAP_OVERFIT = 0.5
GRAD_NORM_LOW = 0.1
GRAD_NORM_HIGH = 0.5
GRAD_NORM_EXPLODING = 2.0


@dataclass
class Health:
    """A single health verdict shown on the dashboard."""

    metric: str
    status: str  # "healthy" | "warning" | "critical"
    value: float | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_log_history(source: str | dict | list) -> list[dict]:
    """Accept a path, a ``trainer_state.json`` dict, or a raw log list."""
    if isinstance(source, str):
        with open(source) as fh:
            source = json.load(fh)
    if isinstance(source, dict):
        return source.get("log_history", [])
    if isinstance(source, list):
        return source
    raise ValueError("Unsupported metrics source")


def _series(log_history: list[dict], key: str) -> list[dict]:
    """Extract ``[{step, epoch, value}]`` rows where ``key`` is present."""
    out = []
    for row in log_history:
        if key in row and row[key] is not None:
            out.append(
                {
                    "step": row.get("step"),
                    "epoch": round(row.get("epoch", 0.0), 4),
                    "value": float(row[key]),
                }
            )
    return out


def analyze(source: str | dict | list) -> dict[str, Any]:
    """Return a complete analysis payload consumed by the web dashboard."""
    log_history = load_log_history(source)

    train_loss = _series(log_history, "loss")
    eval_loss = _series(log_history, "eval_loss")
    grad_norm = _series(log_history, "grad_norm")
    lr = _series(log_history, "learning_rate")

    loss_gap = _loss_gap(train_loss, eval_loss)
    best = _best_checkpoint(eval_loss)
    health = _health_checks(train_loss, eval_loss, loss_gap, grad_norm)
    summary = _summary(train_loss, eval_loss, grad_norm, best, log_history)

    return {
        "series": {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "grad_norm": grad_norm,
            "learning_rate": lr,
            "loss_gap": loss_gap,
        },
        "best_checkpoint": best,
        "health": [h.to_dict() for h in health],
        "summary": summary,
    }


def _loss_gap(train_loss: list[dict], eval_loss: list[dict]) -> list[dict]:
    """train-loss minus eval-loss, aligned at each eval epoch.

    We compare each eval point against the nearest preceding train-loss log so
    the gap reflects roughly the same moment in training.
    """
    if not eval_loss or not train_loss:
        return []
    gaps = []
    for ev in eval_loss:
        nearest = None
        for tr in train_loss:
            if tr["epoch"] <= ev["epoch"] + 1e-6:
                nearest = tr
            else:
                break
        if nearest is None:
            nearest = train_loss[0]
        gaps.append(
            {
                "epoch": ev["epoch"],
                "step": ev["step"],
                "value": round(ev["value"] - nearest["value"], 4),
            }
        )
    return gaps


def _best_checkpoint(eval_loss: list[dict]) -> dict | None:
    """Lowest eval-loss point — the checkpoint to keep (load_best_model_at_end)."""
    if not eval_loss:
        return None
    best = min(eval_loss, key=lambda r: r["value"])
    return {"epoch": best["epoch"], "step": best["step"], "eval_loss": best["value"]}


def _health_checks(
    train_loss: list[dict],
    eval_loss: list[dict],
    loss_gap: list[dict],
    grad_norm: list[dict],
) -> list[Health]:
    checks: list[Health] = []

    # --- Train loss trend -------------------------------------------------
    if len(train_loss) >= 2:
        first, last = train_loss[0]["value"], train_loss[-1]["value"]
        if last < first:
            checks.append(
                Health(
                    "Train loss",
                    "healthy",
                    round(last, 4),
                    f"Decreasing steadily ({first:.3f} → {last:.3f}).",
                )
            )
        else:
            checks.append(
                Health(
                    "Train loss",
                    "warning",
                    round(last, 4),
                    f"Flat or rising ({first:.3f} → {last:.3f}) — check LR / data.",
                )
            )

    # --- Eval loss: rising while train falls = overfit --------------------
    if len(eval_loss) >= 2:
        ev_first, ev_last = eval_loss[0]["value"], eval_loss[-1]["value"]
        tr_falling = (
            len(train_loss) >= 2 and train_loss[-1]["value"] < train_loss[0]["value"]
        )
        if ev_last <= ev_first:
            checks.append(
                Health(
                    "Eval loss",
                    "healthy",
                    round(ev_last, 4),
                    f"Decreasing with train ({ev_first:.3f} → {ev_last:.3f}).",
                )
            )
        elif tr_falling:
            checks.append(
                Health(
                    "Eval loss",
                    "critical",
                    round(ev_last, 4),
                    "Rising while train falls — overfitting. Use the best checkpoint.",
                )
            )
        else:
            checks.append(
                Health(
                    "Eval loss",
                    "warning",
                    round(ev_last, 4),
                    f"Rising ({ev_first:.3f} → {ev_last:.3f}).",
                )
            )

    # --- Loss gap (overfit magnitude) -------------------------------------
    if loss_gap:
        worst = max(abs(g["value"]) for g in loss_gap)
        if worst < LOSS_GAP_HEALTHY:
            status, msg = "healthy", f"Gap {worst:.3f} < {LOSS_GAP_HEALTHY}."
        elif worst < LOSS_GAP_OVERFIT:
            status, msg = "warning", f"Gap {worst:.3f} approaching overfit zone."
        else:
            status, msg = "critical", f"Gap {worst:.3f} > {LOSS_GAP_OVERFIT} — overfit."
        checks.append(Health("Train/Eval gap", status, round(worst, 4), msg))

    # --- Gradient norm ----------------------------------------------------
    if grad_norm:
        values = [g["value"] for g in grad_norm]
        peak = max(values)
        last = values[-1]
        if peak > GRAD_NORM_EXPLODING:
            checks.append(
                Health(
                    "Gradient norm",
                    "critical",
                    round(peak, 4),
                    f"Peak {peak:.2f} > {GRAD_NORM_EXPLODING} — exploding, reduce LR.",
                )
            )
        elif last > GRAD_NORM_HIGH:
            checks.append(
                Health(
                    "Gradient norm",
                    "warning",
                    round(last, 4),
                    f"Elevated ({last:.2f}); healthy band is "
                    f"{GRAD_NORM_LOW}-{GRAD_NORM_HIGH}.",
                )
            )
        else:
            checks.append(
                Health(
                    "Gradient norm",
                    "healthy",
                    round(last, 4),
                    f"Stable in {GRAD_NORM_LOW}-{GRAD_NORM_HIGH} band.",
                )
            )

    return checks


def _summary(
    train_loss: list[dict],
    eval_loss: list[dict],
    grad_norm: list[dict],
    best: dict | None,
    log_history: list[dict],
) -> dict[str, Any]:
    epochs = max((r.get("epoch", 0) for r in log_history), default=0)
    steps = max((r.get("step", 0) for r in log_history), default=0)
    return {
        "total_epochs": round(epochs, 2),
        "total_steps": steps,
        "final_train_loss": round(train_loss[-1]["value"], 4) if train_loss else None,
        "final_eval_loss": round(eval_loss[-1]["value"], 4) if eval_loss else None,
        "best_eval_loss": best["eval_loss"] if best else None,
        "best_epoch": best["epoch"] if best else None,
        "peak_grad_norm": round(max((g["value"] for g in grad_norm), default=0), 4),
        "num_eval_points": len(eval_loss),
        "num_train_points": len(train_loss),
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/trainer_state.json"
    if os.path.exists(path):
        print(json.dumps(analyze(path)["summary"], indent=2))
    else:
        print(f"No metrics at {path}")
