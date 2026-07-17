"""Fine-tune the SLG question router as a sequence classifier."""

import json
import os
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
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

    tokenizer = AutoTokenizer.from_pretrained(model_to_tune, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_to_tune,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    ).to(torch.device("cuda"))
    _set_pad_token(tokenizer, model)

    training_config = CONFIG["training"]
    data_config = CONFIG["data"]
    test_split_ratio = data_config["test_split_ratio"]
    max_length = data_config["max_length"]

    dataset = Dataset.from_list(rows).train_test_split(
        test_size=test_split_ratio,
        seed=int(CONFIG["seed"]),
    )

    def tokenize_function(example):
        return tokenizer(
            example["question"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(["question"])

    lora_config = training_config["lora"]
    peft_params = LoraConfig(
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        r=lora_config["r"],
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

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=training_config["num_epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=training_config["logging_steps"],
        seed=int(CONFIG["seed"]),
        fp16=True,
        use_cpu=False,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        log_level="info",
        logging_dir=logging_dir,
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=training_config["max_grad_norm"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        optim="adamw_torch",
        label_smoothing_factor=training_config["label_smoothing_factor"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=training_config["save_total_limit"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"]
            )
        ],
    )

    trainer.train()
    trainer.model.to(torch.device("cuda"))
    trainer.evaluate()

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
    }
    with open(os.path.join(save_path, ROUTER_METADATA_FILE), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    training_log_path = os.path.join(save_path, "training_log.txt")
    with open(training_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(str(trainer.state.log_history))
