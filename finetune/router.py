"""Fine-tune the SLG question router as a sequence classifier."""

import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from config import CONFIG
from logging_config import logger
from utils.path_utils import ensure_dir, slg_expert_id_from_filename


ROUTER_METADATA_FILE = "router_metadata.json"


def _collect_router_rows(
    split_by_title_dir: str,
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    split_files = sorted(
        file for file in os.listdir(split_by_title_dir) if file.endswith(".json")
    )
    if not split_files:
        raise ValueError(
            f"No JSON files found in {split_by_title_dir}; cannot train SLG router."
        )

    expert_ids = [slg_expert_id_from_filename(file) for file in split_files]
    label2id = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}

    rows: List[Dict[str, object]] = []
    for file, expert_id in zip(split_files, expert_ids):
        data_path = os.path.join(split_by_title_dir, file)
        with open(data_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        for example in examples:
            question = str(example.get("question", "")).strip()
            if question:
                rows.append({"question": question, "labels": label2id[expert_id]})

    if len(label2id) < 2:
        raise ValueError("SLG router requires at least two expert classes.")
    if len(rows) < 2:
        raise ValueError("SLG router requires at least two labeled questions.")

    return rows, label2id


def _expert_id_from_title(title: str) -> str:
    split_title = (
        str(title)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\n", "_")
        .lower()
    )
    return slg_expert_id_from_filename(f"{split_title}.json")


def _collect_router_test_rows(
    qa_test_path: str,
    label2id: Dict[str, int],
) -> List[Dict[str, object]]:
    with open(qa_test_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    rows: List[Dict[str, object]] = []
    skipped = 0
    for example in examples:
        question = str(example.get("question", "")).strip()
        expert_id = _expert_id_from_title(example.get("title", ""))
        if not question or expert_id not in label2id:
            skipped += 1
            continue
        rows.append({"question": question, "labels": label2id[expert_id]})

    if skipped:
        logger.warning(
            "Skipped %s router test rows whose title did not map to a trained expert.",
            skipped,
        )
    if not rows:
        raise ValueError(
            f"No qa_test rows in {qa_test_path} map to trained router classes."
        )
    return rows


def _set_pad_token(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
) -> None:
    if tokenizer.pad_token is None:
        reserved_pad = "<|reserved_special_token_15|>"
        if reserved_pad in tokenizer.get_vocab():
            tokenizer.pad_token = reserved_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def _split_router_rows(
    rows: List[Dict[str, object]],
    test_split_ratio: float,
    seed: int | None,
) -> DatasetDict:
    """Create train/eval splits from the already-prepared router training data."""
    if not 0 < test_split_ratio < 0.5:
        raise ValueError(
            f"Router test_split_ratio must be in (0, 0.5), got {test_split_ratio}."
        )

    labels = [int(row["labels"]) for row in rows]
    indices = list(range(len(rows)))

    train_idx, eval_idx = train_test_split(
        indices,
        test_size=test_split_ratio,
        random_state=seed,
        stratify=labels,
    )

    return DatasetDict({
        "train": Dataset.from_list([rows[i] for i in train_idx]),
        "eval": Dataset.from_list([rows[i] for i in eval_idx]),
    })


def _compute_accuracy(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(predictions == labels))}


class RouterMetricsLoggerCallback(TrainerCallback):
    """Mirror router accuracy metrics into the project logger."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if "eval_accuracy" in metrics:
            logger.info(
                "SLG router eval accuracy at step %s: %.4f",
                state.global_step,
                float(metrics["eval_accuracy"]),
            )


def finetune_slg_router(
    model_to_tune: str,
    adapter_name: str,
    split_by_title_dir: str,
    experiment_number: str,
) -> None:
    """Fine-tune Llama 3.2 1B Instruct as question -> SLG expert classifier."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found! Please ensure you have a CUDA-compatible GPU.")

    rows, label2id = _collect_router_rows(split_by_title_dir)
    id2label = {idx: label for label, idx in label2id.items()}

    logger.info(
        "Training SLG router on %s questions across %s experts.",
        len(rows),
        len(label2id),
    )

    base_training_config = CONFIG["training"]
    router_config = CONFIG.get("slg_router_finetuning", {})
    training_config = router_config.get("training", {})
    data_config = CONFIG["data"]

    def training_value(key: str, fallback_key: str = None, default=None):
        if key in training_config:
            return training_config[key]
        source_key = fallback_key or key
        return base_training_config.get(source_key, default)

    requested_fp16 = bool(training_value("fp16", default=True))
    bf16_supported = torch.cuda.is_bf16_supported()
    use_bf16 = requested_fp16 and bf16_supported
    use_fp16 = requested_fp16 and not use_bf16
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32
    logger.info(
        "Using %s precision for SLG router training.",
        "bf16" if use_bf16 else "fp16 AMP with fp32 model weights" if use_fp16 else "fp32",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_to_tune, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_to_tune,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=model_dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(torch.device("cuda"))
    _set_pad_token(tokenizer, model)

    test_split_ratio = data_config["test_split_ratio"]
    max_length = int(training_value("max_length", default=data_config["max_length"]))

    seed_env = os.getenv("SEED")
    configured_seed = (
        int(seed_env)
        if seed_env is not None
        else training_config.get("seed", CONFIG.get("seed"))
    )
    split_seed = None if configured_seed is None else int(configured_seed)
    trainer_seed = (
        random.SystemRandom().randint(0, 2**32 - 1)
        if configured_seed is None
        else int(configured_seed)
    )
    dataset = _split_router_rows(
        rows=rows,
        test_split_ratio=float(test_split_ratio),
        seed=split_seed,
    )
    test_rows = _collect_router_test_rows(CONFIG["files"]["qa_test"], label2id)
    dataset["test"] = Dataset.from_list(test_rows)

    def tokenize_function(example):
        return tokenizer(
            example["question"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(["question"])

    base_lora_config = base_training_config["lora"]
    lora_config = router_config.get("lora", {})
    peft_params = LoraConfig(
        lora_alpha=lora_config.get(
            "lora_alpha",
            lora_config.get("alpha", base_lora_config["alpha"]),
        ),
        lora_dropout=lora_config.get(
            "lora_dropout",
            lora_config.get("dropout", base_lora_config["dropout"]),
        ),
        r=lora_config.get("r", base_lora_config["r"]),
        target_modules=lora_config.get("target_modules"),
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_params)

    paths_config = CONFIG["paths"]
    checkpoint_dir = os.path.join(
        paths_config["checkpoints"],
        experiment_number,
        adapter_name,
    )
    ensure_dir(checkpoint_dir)

    logging_dir = os.path.join(CONFIG["logging"]["log_dir"], experiment_number)

    per_device_train_batch_size = int(
        training_value("per_device_train_batch_size")
    )
    gradient_accumulation_steps = training_value("gradient_accumulation_steps")
    effective_batch_size = training_config.get("effective_batch_size")
    if gradient_accumulation_steps is None:
        if effective_batch_size is None:
            gradient_accumulation_steps = base_training_config[
                "gradient_accumulation_steps"
            ]
        else:
            effective_batch_size = int(effective_batch_size)
            if effective_batch_size % per_device_train_batch_size != 0:
                raise ValueError(
                    "Router effective_batch_size must be divisible by "
                    "per_device_train_batch_size."
                )
            gradient_accumulation_steps = max(
                1, effective_batch_size // per_device_train_batch_size
            )
    gradient_accumulation_steps = int(gradient_accumulation_steps)

    eval_strategy = training_value("eval_strategy", default="epoch")
    save_strategy = training_value("save_strategy", default="epoch")
    eval_steps = training_config.get("eval_steps")
    save_steps = training_config.get("save_steps")
    if save_steps is None:
        save_steps = eval_steps

    training_args_kwargs = dict(
        output_dir=checkpoint_dir,
        num_train_epochs=training_value("num_train_epochs", "num_epochs"),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=training_value("logging_steps"),
        seed=trainer_seed,
        fp16=use_fp16,
        bf16=use_bf16,
        use_cpu=False,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        log_level="info",
        logging_dir=logging_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=training_value("per_device_eval_batch_size"),
        learning_rate=training_value("learning_rate"),
        weight_decay=training_value("weight_decay", default=0.0),
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=training_value("max_grad_norm", default=1.0),
        warmup_ratio=training_value("warmup_ratio", default=0.0),
        lr_scheduler_type=training_value("lr_scheduler_type", default="cosine"),
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=training_value("optim", default="adamw_torch"),
        label_smoothing_factor=training_value(
            "label_smoothing_factor",
            default=0.0,
        ),
        load_best_model_at_end=training_value("load_best_model_at_end", default=True),
        metric_for_best_model=training_value("metric_for_best_model", default="eval_loss"),
        greater_is_better=training_value("greater_is_better", default=False),
        save_total_limit=training_value("save_total_limit"),
    )
    if eval_steps is not None:
        training_args_kwargs["eval_steps"] = int(eval_steps)
    if save_steps is not None:
        training_args_kwargs["save_steps"] = int(save_steps)
    if training_config.get("warmup_steps") is not None:
        training_args_kwargs["warmup_steps"] = int(training_config["warmup_steps"])

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_value("early_stopping_patience"),
                early_stopping_threshold=training_value(
                    "early_stopping_threshold",
                    default=0.0,
                ),
            ),
            RouterMetricsLoggerCallback(),
        ],
        compute_metrics=_compute_accuracy,
    )

    trainer.train()
    trainer.model.to(torch.device("cuda"))
    eval_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["eval"],
        metric_key_prefix="eval",
    )
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
        metric_key_prefix="test",
    )
    logger.info("SLG router final eval metrics: %s", eval_metrics)
    logger.info("SLG router final test metrics on qa_test: %s", test_metrics)

    save_path = os.path.join(
        paths_config["experiments"],
        experiment_number,
        adapter_name,
    )
    ensure_dir(save_path)
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metadata = {
        "base_model": model_to_tune,
        "adapter_name": adapter_name,
        "label2id": label2id,
        "id2label": {str(idx): label for idx, label in id2label.items()},
        "router_finetuning": router_config,
        "effective_gradient_accumulation_steps": gradient_accumulation_steps,
        "trainer_seed": trainer_seed,
    }
    with open(os.path.join(save_path, ROUTER_METADATA_FILE), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    training_log_path = os.path.join(save_path, "training_log.txt")
    with open(training_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(str(trainer.state.log_history))
        log_file.write("\n\nFinal eval metrics:\n")
        log_file.write(json.dumps(eval_metrics, indent=2))
        log_file.write("\n\nFinal test metrics on qa_test:\n")
        log_file.write(json.dumps(test_metrics, indent=2))
