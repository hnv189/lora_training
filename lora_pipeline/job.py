"""Background training-job manager for the dashboard.

Launches training as a subprocess, tracks its lifecycle, and exposes a live
metrics view by reading the ``trainer_state.json`` the job writes incrementally.

Two execution modes:

* ``real``     — runs ``python -m lora_pipeline.train`` (needs GPU + ML wheels).
* ``simulate`` — runs ``scripts/simulate_training.py``, which writes a realistic
                 ``trainer_state.json`` step-by-step so the live loss monitoring
                 works on any machine (no GPU required).

``auto`` (the default) picks ``real`` when CUDA is available, else ``simulate``.

State is kept in ``runs/current/job.json`` so status survives a page reload, and
a watcher thread flips the job to completed/failed when the process exits.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

from . import metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
CURRENT_RUN = os.path.join(RUNS_DIR, "current")
JOB_FILE = os.path.join(CURRENT_RUN, "job.json")
STATE_FILE = os.path.join(CURRENT_RUN, "trainer_state.json")
LOG_FILE = os.path.join(CURRENT_RUN, "train.log")

# Live handle to the running process (lost on Flask restart; pid liveness covers
# that case for status reporting).
_proc: subprocess.Popen | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def training_deps_available() -> bool:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401

        return True
    except Exception:
        return False


def resolve_mode(requested: str) -> str:
    if requested in ("real", "simulate"):
        return requested
    # auto
    return "real" if gpu_available() and training_deps_available() else "simulate"


def _read_job() -> dict[str, Any]:
    if os.path.exists(JOB_FILE):
        try:
            with open(JOB_FILE) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"status": "idle"}


def _write_job(job: dict[str, Any]) -> None:
    os.makedirs(CURRENT_RUN, exist_ok=True)
    with open(JOB_FILE, "w") as fh:
        json.dump(job, fh, indent=2)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start(config: dict[str, Any]) -> dict[str, Any]:
    """Start a training job. Raises RuntimeError if one is already running."""
    global _proc
    with _lock:
        existing = _read_job()
        if existing.get("status") == "running" and _pid_alive(existing.get("pid")):
            raise RuntimeError("A training job is already running.")

        os.makedirs(CURRENT_RUN, exist_ok=True)
        # Clear previous run artifacts so the live chart starts fresh.
        for path in (STATE_FILE, LOG_FILE):
            if os.path.exists(path):
                os.remove(path)

        mode = resolve_mode(config.get("mode", "auto"))
        epochs = int(config.get("epochs", 3))
        rank = int(config.get("rank", 32))
        lr = float(config.get("learning_rate", 2e-4))
        dataset = config.get("dataset", "./dataset")
        step_delay = float(config.get("step_delay", 0.4))

        cmd = _build_command(mode, epochs, rank, lr, dataset, step_delay)

        log_fh = open(LOG_FILE, "w")
        _proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group so we can stop cleanly
        )

        job = {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "status": "running",
            "mode": mode,
            "pid": _proc.pid,
            "config": {
                "epochs": epochs,
                "rank": rank,
                "learning_rate": lr,
                "dataset": dataset,
            },
            "started_at": _now(),
            "ended_at": None,
            "error": None,
        }
        _write_job(job)

        threading.Thread(target=_watch, args=(_proc, log_fh), daemon=True).start()
        return job


def _build_command(
    mode: str, epochs: int, rank: int, lr: float, dataset: str, step_delay: float
) -> list[str]:
    if mode == "real":
        return [
            sys.executable, "-m", "lora_pipeline.train",
            "--dataset", dataset,
            "--output", CURRENT_RUN,
            "--epochs", str(epochs),
            "--rank", str(rank),
        ]
    return [
        sys.executable, os.path.join(BASE_DIR, "scripts", "simulate_training.py"),
        "--output", CURRENT_RUN,
        "--epochs", str(epochs),
        "--rank", str(rank),
        "--learning-rate", str(lr),
        "--step-delay", str(step_delay),
    ]


def _watch(proc: subprocess.Popen, log_fh) -> None:
    """Wait for the process to finish and record the terminal status."""
    code = proc.wait()
    try:
        log_fh.close()
    except Exception:
        pass
    job = _read_job()
    if job.get("status") == "stopped":
        pass  # user-initiated stop already recorded
    elif code == 0:
        job["status"] = "completed"
    else:
        job["status"] = "failed"
        job["error"] = f"process exited with code {code}"
    job["ended_at"] = _now()
    _write_job(job)


def stop() -> dict[str, Any]:
    """Terminate the running job's process group."""
    global _proc
    with _lock:
        job = _read_job()
        pid = job.get("pid")
        if job.get("status") != "running" or not _pid_alive(pid):
            return {"status": job.get("status", "idle"), "message": "no running job"}
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        job["status"] = "stopped"
        job["ended_at"] = _now()
        _write_job(job)
        return job


def status(log_lines: int = 40) -> dict[str, Any]:
    """Return job status, live metrics analysis, and a tail of the log."""
    job = _read_job()

    # Reconcile a stale "running" whose process has died (e.g. Flask restarted).
    if job.get("status") == "running" and not _pid_alive(job.get("pid")):
        job["status"] = "completed" if os.path.exists(STATE_FILE) else "failed"
        job["ended_at"] = job.get("ended_at") or _now()
        _write_job(job)

    payload: dict[str, Any] = {"job": job, "metrics": None, "log": ""}

    if os.path.exists(STATE_FILE):
        try:
            payload["metrics"] = metrics.analyze(STATE_FILE)
        except Exception as exc:
            payload["metrics_error"] = str(exc)

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE) as fh:
                lines = fh.readlines()
            payload["log"] = "".join(lines[-log_lines:])
        except Exception:
            pass

    payload["environment"] = {
        "gpu_available": gpu_available(),
        "training_deps_available": training_deps_available(),
    }
    return payload


def current_state_path() -> str | None:
    """Path to the active run's trainer_state.json, if it exists."""
    return STATE_FILE if os.path.exists(STATE_FILE) else None
