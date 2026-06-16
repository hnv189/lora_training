"""Phases 2-3 — Model + LoRA setup and training.

This is the runnable training entrypoint for the RTX 2000 Ada (16GB) target.
It is import-safe: the heavy ML libraries are only imported inside functions so
the web app and the rest of the pipeline run without a GPU or CUDA wheels
installed.

Run with:  python -m lora_pipeline.train --dataset ./dataset
"""

from __future__ import annotations

import argparse

BASE_MODEL = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are a code implementation expert. "
    "When given a formal definition, you analyse its intent and constraints, "
    "then write a complete, idiomatic implementation. /no_think"
)

# LoRA config — targets attention AND FFN. Attention teaches *what to look at*;
# FFN (gate/up/down_proj) teaches *how to reason about it* (plan 2.2).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",          # FFN / reasoning
]


def build_lora_config(r: int = 32, alpha: int = 64, dropout: float = 0.05):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_4bit(base_model: str = BASE_MODEL):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_training_args(output_dir: str = "./lora-checkpoints", epochs: int = 3):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 8
        gradient_checkpointing=True,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
    )


def make_live_metrics_callback(output_dir: str):
    """A TrainerCallback that dumps trainer_state.json on every log/eval.

    HF Trainer only persists trainer_state.json at checkpoint-save time, which
    is too coarse for live dashboard monitoring. This writes it (atomically) on
    every logging and evaluation step so the dashboard updates in real time.
    """
    import json as _json
    import os as _os
    from transformers import TrainerCallback

    state_path = _os.path.join(output_dir, "trainer_state.json")

    class LiveMetricsCallback(TrainerCallback):
        def _dump(self, state):
            _os.makedirs(output_dir, exist_ok=True)
            payload = {
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
                "epoch": state.epoch,
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "num_train_epochs": state.num_train_epochs,
                "log_history": state.log_history,
            }
            tmp = state_path + ".tmp"
            with open(tmp, "w") as fh:
                _json.dump(payload, fh, indent=2)
            _os.replace(tmp, state_path)

        def on_log(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self._dump(state)

        def on_evaluate(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self._dump(state)

    return LiveMetricsCallback()


def train(dataset_dir: str = "./dataset", output_dir: str = "./lora-checkpoints",
          epochs: int = 3, r: int = 32, max_seq_length: int = 2048):
    from datasets import load_from_disk
    from peft import get_peft_model
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from .prepare_data import RESPONSE_TEMPLATE

    model, tokenizer = load_model_4bit()
    model = get_peft_model(model, build_lora_config(r=r))
    model.print_trainable_parameters()  # expect ~1-2% of params trainable

    split = load_from_disk(dataset_dir)

    # Loss masking — only train on tokens after "### Implementation:" so the
    # model never wastes capacity learning to reproduce the prompt (plan 3.2).
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=build_training_args(output_dir, epochs),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )
    # Live dashboard monitoring: persist trainer_state.json on every log/eval.
    trainer.add_callback(make_live_metrics_callback(output_dir))
    trainer.train()
    trainer.save_model(f"{output_dir}/best")

    # The trainer writes trainer_state.json into output_dir — the web dashboard
    # reads it straight from there for live loss analysis.
    print(f"Done. Point the dashboard at {output_dir}/trainer_state.json")
    return trainer


def main():
    ap = argparse.ArgumentParser(description="LoRA training (Qwen3-8B, 4-bit)")
    ap.add_argument("--dataset", default="./dataset")
    ap.add_argument("--output", default="./lora-checkpoints")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()
    train(args.dataset, args.output, args.epochs, args.rank, args.max_seq_length)


if __name__ == "__main__":
    main()
