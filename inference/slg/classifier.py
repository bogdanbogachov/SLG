"""(Router) Llama-3.2-1B sequence-classification head that picks the expert.

This is the routing **decider**, replacing the 8B reasoning router. Diagnostics
on a real run showed the reasoning router degraded routing from ~85% (cosine
shortlist top-1) down to ~45% by second-guessing a shortlist that already
contained the correct expert 99.4% of the time. A small discriminative
classifier fine-tuned on the training questions maps a question straight to an
expert and restores routing accuracy; the 8B reasoner is kept **only** as a
tiebreaker for genuinely ambiguous questions (small top1-top2 margin).

Trained once over the **full** expert set (``--finetune_router``) and cached at
``experiments/<exp>/slg_router/``. At query time the output distribution is
restricted to the allowed pool via a mask, so the scalability sweep reuses one
router across pool sizes with no retraining — the same allow-list-at-query-time
pattern the retriever uses. When no trained router exists, the pipeline falls
back to the cosine retriever (see ``ExpertRetriever.scores``).
"""

import json
import os
from typing import Dict, List, Optional, Set

import numpy as np

from config import CONFIG
from logging_config import logger
from utils.path_utils import ensure_dir, get_slg_router_dir


_LABELS_FILE = "labels.json"


def _router_cfg() -> dict:
    return CONFIG["routing"].get("router", {})


def _base_model_path() -> str:
    paths_cfg = CONFIG["paths"]
    return os.path.join(paths_cfg["downloaded_models"], paths_cfg["models"]["3_2_1b"])


class ExpertClassifier:
    """Fine-tuned Llama-1B classification head; maps a question to an expert.

    Lazily loads the model. ``available`` reports whether a trained router exists
    for the experiment; when it does not the caller should fall back to cosine
    ranking. The allow-list (``allowed_experts``) is applied by the pipeline when
    it reads the returned score dict, so one router serves every pool subset.
    """

    def __init__(self, experiment: str, allowed_experts: Optional[Set[str]] = None):
        self.experiment = experiment
        self._router_dir = get_slg_router_dir(experiment)
        self._allowed = set(allowed_experts) if allowed_experts is not None else None
        self._max_length = int(_router_cfg().get("max_length", 256))
        self._batch_size = int(_router_cfg().get("batch_size", 16))

        self.labels: List[str] = []
        labels_path = os.path.join(self._router_dir, _LABELS_FILE)
        adapter_cfg = os.path.join(self._router_dir, "adapter_config.json")
        self._available = os.path.isfile(labels_path) and os.path.isfile(adapter_cfg)
        if self._available:
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = list(json.load(f))

        self._model = None
        self._tokenizer = None

    @property
    def available(self) -> bool:
        return self._available

    # ----------------------------------------------------------- lifecycle
    def load(self) -> "ExpertClassifier":
        if self._model is not None:
            return self
        if not self._available:
            raise RuntimeError(
                f"No trained router at {self._router_dir}. Run --finetune_router first."
            )
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the router classifier.")
        device = torch.device("cuda")

        self._tokenizer = AutoTokenizer.from_pretrained(self._router_dir, trust_remote_code=True)
        base = AutoModelForSequenceClassification.from_pretrained(
            _base_model_path(),
            num_labels=len(self.labels),
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        base.config.pad_token_id = self._tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, self._router_dir)
        self._model = model.to(device).eval()
        logger.info("Loaded router classifier (%d experts) from %s", len(self.labels), self._router_dir)
        return self

    def unload(self) -> None:
        if self._model is not None:
            from utils.model_loader import cleanup_model_memory

            cleanup_model_memory(self._model, self._tokenizer)
            self._model = None
            self._tokenizer = None

    # ------------------------------------------------------------- predict
    def predict_proba(self, questions: List[str]) -> np.ndarray:
        """Softmax class probabilities, shape ``[len(questions), len(labels)]``.

        Columns are ordered by ``self.labels``. The allow-list is NOT applied
        here — the caller masks to its pool when reading the scores.
        """
        if not questions:
            return np.empty((0, len(self.labels)), dtype="float32")
        import torch

        self.load()
        out = np.empty((len(questions), len(self.labels)), dtype="float32")
        for start in range(0, len(questions), self._batch_size):
            batch = questions[start : start + self._batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="pt",
            ).to(self._model.device)
            with torch.no_grad():
                logits = self._model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out[start : start + len(batch)] = probs
        return out

    def logits_batch(self, questions: List[str]) -> List[Dict[str, float]]:
        """Per-question ``{expert_id: raw logit}`` over the allowed pool (NO softmax).

        The pipeline softmaxes over the actual candidate set (allowed minus the
        experts already tried this question) *after* masking, so the reject floor
        and tie test always see a proper probability distribution over the real
        candidates — not post-softmax mass stranded on a masked-out expert.
        """
        import torch

        if not questions:
            return []
        self.load()
        result: List[Dict[str, float]] = []
        for start in range(0, len(questions), self._batch_size):
            batch = questions[start : start + self._batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="pt",
            ).to(self._model.device)
            with torch.no_grad():
                logits = self._model(**enc).logits.float().cpu().numpy()
            for row in logits:
                result.append({
                    eid: float(row[j])
                    for j, eid in enumerate(self.labels)
                    if self._allowed is None or eid in self._allowed
                })
        return result


# ============================================================== training
def _slug_title(title: str) -> str:
    """Ground-truth expert id for a title (mirrors split_qa_pairs_by_title)."""
    return (title or "").replace(" ", "_").replace("/", "_").replace("\n", "_").lower()


def _expert_labels(split_dir: str) -> List[str]:
    """Canonical, sorted expert ids = the split_by_title file stems (== adapter names)."""
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(split_dir) if f.endswith(".json")
    )


def train_router(experiment: str) -> str:
    """Fine-tune the Llama-1B router classifier over the full expert set.

    Samples a seeded, class-balanced subset of ``qa_train`` (``train_per_class``
    questions per expert), fine-tunes a LoRA sequence-classification head, and
    saves the adapter + tokenizer + ``labels.json`` to ``experiments/<exp>/slg_router/``.
    Returns the output directory.
    """
    import numpy as _np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train the router classifier.")

    cfg = _router_cfg()
    seed = int(CONFIG["seed"])
    per_class = int(cfg.get("train_per_class", 400))
    max_length = int(cfg.get("max_length", 256))
    split_dir = CONFIG["paths"]["split_by_title"]

    labels = _expert_labels(split_dir)
    label2id = {e: i for i, e in enumerate(labels)}
    logger.info("Router training over %d experts: %s", len(labels), labels)

    with open(CONFIG["files"]["qa_train"], "r", encoding="utf-8") as f:
        data = json.load(f)

    # Seeded, class-balanced subsample: fast to train and enough to hit the ceiling.
    rng = _np.random.default_rng(seed)
    by_expert: Dict[str, List[str]] = {}
    for item in data:
        eid = _slug_title(item.get("title"))
        if eid in label2id:
            by_expert.setdefault(eid, []).append(item["question"])
    questions: List[str] = []
    y: List[int] = []
    for eid, qs in by_expert.items():
        idx = rng.permutation(len(qs))[:per_class]
        for j in idx:
            questions.append(qs[j])
            y.append(label2id[eid])
    logger.info("Router training set: %d questions (<= %d/class).", len(questions), per_class)

    tokenizer = AutoTokenizer.from_pretrained(_base_model_path(), trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Distinct reserved pad token (not eos) so last-token pooling is correct.
        tokenizer.pad_token = "<|reserved_special_token_15|>"

    ds = Dataset.from_dict({"text": questions, "label": y})
    ds = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length),
        batched=True,
    ).train_test_split(test_size=0.1, seed=seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        _base_model_path(),
        num_labels=len(labels),
        id2label={i: e for e, i in label2id.items()},
        label2id=label2id,
        # Load in fp32 for training (the ONE deviation from the expert/baseline
        # recipe): a freshly-initialised classification head in fp16 with fp16
        # master weights underflows and never learns (held-out accuracy stuck at
        # chance). The experts have no random head, so they load fp16 fine. Mixed
        # precision here is still fp16 autocast over fp32 master weights.
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    peft_cfg = LoraConfig(
        task_type="SEQ_CLS",
        r=int(CONFIG["training"]["lora"]["r"]),
        lora_alpha=int(CONFIG["training"]["lora"]["alpha"]),
        lora_dropout=float(CONFIG["training"]["lora"]["dropout"]),
        target_modules=CONFIG["training"]["lora"].get("target_modules"),
        modules_to_save=["score"],  # keep + save the classification head
    )
    model = get_peft_model(model, peft_cfg)
    model = model.to("cuda")

    from transformers import DataCollatorWithPadding

    def compute_metrics(eval_pred):
        preds = _np.argmax(eval_pred.predictions, axis=-1)
        return {"accuracy": float((preds == eval_pred.label_ids).mean())}

    out_dir = get_slg_router_dir(experiment)
    ensure_dir(out_dir)
    # Same training recipe as the experts/baselines (finetune.py): CONFIG['training']
    # drives epochs, LR, schedule, weight decay, warmup, early stopping + best-model
    # selection, and the model-size-aware batch (the router base is the 1B, so it
    # uses the default per_device batch). Only the fp32 weight load differs.
    tc = CONFIG["training"]
    args = TrainingArguments(
        output_dir=os.path.join(CONFIG["paths"]["checkpoints"], experiment, "slg_router"),
        # Router override: a higher epoch cap than the experts; early stopping +
        # load_best_model_at_end still decide when to stop.
        num_train_epochs=int(cfg.get("num_epochs", tc["num_epochs"])),
        learning_rate=float(tc["learning_rate"]),
        per_device_train_batch_size=int(tc["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tc["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tc["gradient_accumulation_steps"]),
        weight_decay=float(tc["weight_decay"]),
        warmup_ratio=float(tc["warmup_ratio"]),
        max_grad_norm=float(tc["max_grad_norm"]),
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        label_smoothing_factor=float(tc["label_smoothing_factor"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=int(tc["save_total_limit"]),
        logging_steps=int(tc["logging_steps"]),
        seed=seed,
        fp16=True,   # fp16 autocast over the fp32 master weights loaded above
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(tc["early_stopping_patience"]))],
    )
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Router held-out accuracy: %.4f", metrics.get("eval_accuracy", float("nan")))

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, _LABELS_FILE), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    logger.info("Saved router classifier to %s", out_dir)
    return out_dir
