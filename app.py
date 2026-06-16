"""LoRA Training Dashboard — web app.

A self-contained Flask app that drives and analyses the def->implement
behaviour-shaping pipeline. The centrepiece is training-data analysis: loss
curves, overfit detection, gradient norms and the behaviour-eval scorecard.

Run:  python app.py   (then open http://localhost:5000)

Every endpoint degrades gracefully without GPU/ML wheels installed — the
analysis modules are pure Python with heuristic fallbacks.
"""

from __future__ import annotations

import io
import json
import os

from flask import Flask, jsonify, render_template, request

from lora_pipeline import data_audit, evaluate, metrics, prepare_data

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
DEFAULT_METRICS = os.path.join(SAMPLE_DIR, "trainer_state.json")
PAIRS_JSON = os.path.join(SAMPLE_DIR, "pairs.json")
EVAL_JSON = os.path.join(SAMPLE_DIR, "behaviour_eval.json")

# In-memory store for an uploaded trainer_state.json (per process).
_uploaded_state: dict | None = None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API — training metrics / loss analysis
# ---------------------------------------------------------------------------


@app.route("/api/metrics")
def api_metrics():
    """Analyse the current trainer_state (uploaded one wins, else sample)."""
    global _uploaded_state
    source = _uploaded_state
    if source is None:
        if not os.path.exists(DEFAULT_METRICS):
            return jsonify({"error": "no metrics available"}), 404
        source = DEFAULT_METRICS
    try:
        payload = metrics.analyze(source)
        payload["source"] = "uploaded" if _uploaded_state else "sample"
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/metrics/upload", methods=["POST"])
def api_metrics_upload():
    """Accept a HuggingFace trainer_state.json upload and analyse it live."""
    global _uploaded_state
    file = request.files.get("file")
    if file is None:
        # Also accept raw JSON in the request body.
        try:
            _uploaded_state = request.get_json(force=True)
        except Exception:
            return jsonify({"error": "no file or JSON provided"}), 400
    else:
        try:
            _uploaded_state = json.load(io.TextIOWrapper(file.stream, encoding="utf-8"))
        except Exception as exc:
            return jsonify({"error": f"invalid JSON: {exc}"}), 400
    try:
        payload = metrics.analyze(_uploaded_state)
        payload["source"] = "uploaded"
        return jsonify(payload)
    except Exception as exc:
        _uploaded_state = None
        return jsonify({"error": str(exc)}), 400


@app.route("/api/metrics/reset", methods=["POST"])
def api_metrics_reset():
    global _uploaded_state
    _uploaded_state = None
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API — data audit (Phase 0)
# ---------------------------------------------------------------------------


@app.route("/api/audit")
def api_audit():
    try:
        result = data_audit.run_full_audit(
            fallback_json=PAIRS_JSON, use_tokenizer=False
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# API — data preparation quality report (Phase 1)
# ---------------------------------------------------------------------------


@app.route("/api/prepare")
def api_prepare():
    try:
        with open(PAIRS_JSON) as fh:
            pairs = json.load(fh)
        report = prepare_data.quality_report(pairs)
        # Include one rendered sample so the UI can show the prompt format.
        report["example_prompt"] = prepare_data.format_sample(pairs[0])["text"]
        report["has_reasoning"] = all(p.get("reasoning") for p in pairs)
        return jsonify(report)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# API — behaviour evaluation (Phase 4)
# ---------------------------------------------------------------------------


@app.route("/api/eval")
def api_eval():
    try:
        with open(EVAL_JSON) as fh:
            samples = json.load(fh)
        return jsonify(evaluate.aggregate_eval(samples))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# API — config snapshot (the plan's training arguments, for the Config tab)
# ---------------------------------------------------------------------------


@app.route("/api/config")
def api_config():
    return jsonify(
        {
            "base_model": "Qwen/Qwen3-8B",
            "hardware": "RTX 2000 Ada (16GB) · 64GB RAM",
            "quantization": "4-bit NF4 + double quant, bf16 compute",
            "lora": {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            },
            "training": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 8,
                "learning_rate": 2e-4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.05,
                "max_grad_norm": 0.3,
                "optim": "paged_adamw_8bit",
                "max_seq_length": 2048,
            },
            "loss_masking": "DataCollatorForCompletionOnlyLM on '### Implementation:'",
        }
    )


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "version": "1.0.0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
