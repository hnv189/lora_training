# LoRA Training Dashboard — Def → Implement Behaviour Shaping

A fully working **web-based app** for the LoRA training plan that fine-tunes
`Qwen/Qwen3-8B` to *behave* like a developer reading a definition and writing an
implementation — not just memorise input→output mappings.

The dashboard's centrepiece is **training-data analysis**: live loss curves,
overfit detection, gradient-norm health, token-length audit and a behaviour
scorecard — all derived from a real HuggingFace `trainer_state.json`.

![phases](https://img.shields.io/badge/phases-0→5-blue) ![model](https://img.shields.io/badge/base-Qwen3--8B-purple) ![hw](https://img.shields.io/badge/GPU-RTX%202000%20Ada%2016GB-green)

---

## Quick start

```bash
pip install -r requirements.txt          # just Flask — installs anywhere
python scripts/generate_sample_metrics.py # produces a sample training run
python app.py                            # http://localhost:5000
```

The app ships with a realistic sample run and sample def/impl pairs, so every
tab is populated immediately. No GPU required to explore the analysis.

---

## What the dashboard shows

| Tab | What it analyses |
|---|---|
| **Loss Analysis** | Train vs eval loss, best checkpoint, train/eval **gap** (overfit if > 0.5), per-metric health verdicts |
| **Training Dynamics** | Gradient norm (healthy 0.1–0.5, exploding > 2.0) and the cosine LR schedule |
| **Data Audit** (Phase 0) | Pair count + readiness, token-length histogram with **P95 → recommended `max_seq_length`**, pattern-category balance, behavioural fingerprint |
| **Data Prep** (Phase 1) | Quality-filter keep/drop report, reasoning-trace presence, rendered behaviour-first prompt |
| **Behaviour Eval** (Phase 4) | Checklist score (>0.75), CodeBLEU (>0.60), **out-of-distribution pass rate**, failure-mode table |
| **Config** | The exact LoRA + training arguments used |

### Analyse your own run

Click **Upload `trainer_state.json`** on the Loss Analysis tab. The file is the
one HuggingFace `Trainer` writes into your `output_dir`. The dashboard re-runs
the full loss / overfit / gradient analysis on it instantly.

---

## The pipeline (`lora_pipeline/`)

Each module maps 1:1 to a plan phase and is runnable standalone:

```bash
python -m lora_pipeline.data_audit      # Phase 0: counts, tokens, categories, fingerprint
python -m lora_pipeline.prepare_data    # Phase 1: prompt format + quality report
python -m lora_pipeline.train --dataset ./dataset   # Phases 2-3: 4-bit Qwen3-8B + LoRA
python -m lora_pipeline.evaluate        # Phase 4: CodeBLEU + behaviour checklist + OOD
python -m lora_pipeline.metrics path/to/trainer_state.json   # loss analysis
```

| Module | Phase | Key design choice |
|---|---|---|
| `data_audit.py` | 0 | Sets `max_seq_length` to **P95** (not max) to avoid OOM; flags imbalance |
| `prepare_data.py` | 1 | **Behaviour-first prompt** with `### Reasoning:`; synthetic traces via Claude; quality filter |
| `train.py` | 2–3 | LoRA on attention **and FFN** (`gate/up/down_proj`); **loss masking** on `### Implementation:` only |
| `evaluate.py` | 4 | Behaviour checklist + CodeBLEU + the all-important OOD test |
| `metrics.py` | — | Turns training logs into overfit / gradient diagnostics |

### Running real training

On the RTX 2000 Ada box:

```bash
pip install -r requirements-train.txt
export ANTHROPIC_API_KEY=...            # for synthetic reasoning traces (optional)
python -m lora_pipeline.train --dataset ./dataset --epochs 3 --rank 32
```

`train.py` writes `lora-checkpoints/trainer_state.json` — upload it to the
dashboard to watch the loss analysis on your real run.

---

## Why this design (the behaviour distinction)

The plan's core thesis is that naive `def → impl` fine-tuning teaches a *lookup
table*. This pipeline encodes the four fixes that teach *behaviour* instead:

1. **Reasoning traces** bridge def→impl so the model learns a thinking process.
2. **Loss masking** flows gradients only through the implementation tokens.
3. **Behaviour eval > loss metrics** — the checklist + OOD test are first-class.
4. **Quality > quantity** — the quality filter drops stubs and vague defs.

The dashboard makes each of these observable: the gap chart catches overfitting,
the OOD pass-rate catches memorisation, the fingerprint defines "your style".

---

## Project layout

```
app.py                      Flask web server (API + dashboard)
lora_pipeline/              Phase 0–4 pipeline modules + metrics engine
templates/index.html        Dashboard UI
static/css, static/js       Styling + Chart.js front-end
sample_data/                Sample pairs, behaviour eval, trainer_state.json
scripts/                    Sample-metrics generator
requirements.txt            Flask (dashboard)
requirements-train.txt      GPU training deps
```
