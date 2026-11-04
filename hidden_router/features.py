"""Per-layer pooled hidden states of questions from the base (un-adapted) Llama-3.2-1B."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from config import CONFIG
from logging_config import logger


def _cfg():
    return CONFIG.get("hidden_router", {})


def base_model_path() -> str:
    paths = CONFIG["paths"]
    return os.path.join(paths["downloaded_models"], paths["models"]["3_2_1b"])


class FeatureExtractor:
    """Mean-pools each layer's hidden states over the question tokens (BOS excluded).

    Output per question: float16 array [n_layers + 1, hidden] (index 0 = embedding layer).
    The input is the raw question text (no chat template), so the representation depends on the
    question only, not on any prompt boilerplate.
    """

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 max_length: int | None = None, batch_size: int | None = None):
        from transformers import AutoModel, AutoTokenizer

        cfg = _cfg()
        self.model_path = model_path or base_model_path()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = int(max_length or cfg.get("max_length", 512))
        self.batch_size = int(batch_size or cfg.get("batch_size", 32))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=dtype).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, questions: Sequence[str], log_every: int = 2000) -> np.ndarray:
        n = len(questions)
        order = sorted(range(n), key=lambda i: len(questions[i]))  # length-sorted batches: little padding
        out = None
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = self.tokenizer([questions[i] for i in idx], return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.max_length).to(self.device)
            hidden = self.model(**batch, output_hidden_states=True).hidden_states  # (L+1) x [B, T, H]
            mask = batch["attention_mask"].clone()
            if mask.shape[1] > 1:
                mask[:, 0] = 0  # drop BOS: an attention sink with very large norm, identical for all questions
            mask = mask.unsqueeze(-1).to(hidden[0].dtype)
            denom = mask.sum(1).clamp(min=1)
            pooled = torch.stack([(h * mask).sum(1) / denom for h in hidden], dim=1)  # [B, L+1, H]
            pooled = pooled.float().cpu().numpy().astype(np.float16)
            if out is None:
                out = np.empty((n,) + pooled.shape[1:], dtype=np.float16)
            out[idx] = pooled
            if log_every and (start // self.batch_size) % max(1, log_every // self.batch_size) == 0:
                logger.info("Hidden-state extraction: %s/%s questions", min(start + len(idx), n), n)
        if out is None:
            raise ValueError("No questions to encode.")
        return out

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FeatureStore:
    """qid -> pooled features cache on disk (features.npy + qids.json), extended on demand."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.features_path = self.dir / "features.npy"
        self.qids_path = self.dir / "qids.json"
        self._features = None
        self._index = {}
        if self.features_path.is_file() and self.qids_path.is_file():
            self._features = np.load(self.features_path, mmap_mode="r")
            qids = json.loads(self.qids_path.read_text(encoding="utf-8"))
            if len(qids) != len(self._features):
                # Interrupted between writing features.npy and qids.json: start the cache again.
                logger.warning("Feature cache in %s is inconsistent (%s qids vs %s rows); rebuilding it.",
                               self.dir, len(qids), len(self._features))
                self._features = None
            else:
                self._index = {q: i for i, q in enumerate(qids)}

    def missing(self, qids: Iterable[str]) -> List[str]:
        return [q for q in dict.fromkeys(qids) if q not in self._index]

    def ensure(self, items: Sequence[Tuple[str, str]], extractor_factory=FeatureExtractor) -> None:
        """items: (qid, question). Extracts and appends any qid not cached yet."""
        question_by_qid = dict(items)
        todo = self.missing(question_by_qid)
        if not todo:
            return
        logger.info("Extracting hidden states for %s new questions into %s", len(todo), self.dir)
        extractor = extractor_factory()
        try:
            new = extractor.encode([question_by_qid[q] for q in todo])
        finally:
            extractor.close()
        old_qids = list(self._index)
        old = np.asarray(self._features) if self._features is not None else None
        merged = new if old is None else np.concatenate([old, new], axis=0)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "features.tmp.npy"
        np.save(tmp, merged)
        tmp.replace(self.features_path)
        self.qids_path.write_text(json.dumps(old_qids + todo), encoding="utf-8")
        self._features = np.load(self.features_path, mmap_mode="r")
        self._index = {q: i for i, q in enumerate(old_qids + todo)}

    def get(self, qids: Sequence[str]) -> np.ndarray:
        missing = self.missing(qids)
        if missing:
            raise KeyError(f"{len(missing)} qids are not in the feature cache {self.dir}")
        return np.asarray(self._features[[self._index[q] for q in qids]])
