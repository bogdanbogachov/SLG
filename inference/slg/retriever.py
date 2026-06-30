"""Cosine pre-filter over experts using local Jina embeddings.

Each expert is represented by the mean of its (deduplicated) training answers,
embedded with the local ``jina-v2-base-en`` SentenceTransformer. At query time
the user question is embedded once and ranked against every expert by cosine
similarity to produce the shortlist handed to the reasoning router.

Embeddings are computed once per experiment and cached under
``experiments/<exp>/slg_index/`` so repeated runs do not re-embed the corpus.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import CONFIG
from logging_config import logger
from utils.path_utils import ensure_dir, get_slg_index_dir


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class ExpertRetriever:
    """Embed experts (mean of training answers) and rank them against a query."""

    def __init__(self, experiment: str, allowed_experts: Optional[Set[str]] = None):
        self.experiment = experiment
        routing_cfg = CONFIG["routing"]
        self._embedding_dim = int(routing_cfg["embedding_dimension"])
        self._embedding_batch_size = int(routing_cfg.get("embedding_batch_size", 100))

        paths_cfg = CONFIG["paths"]
        self._split_dir = paths_cfg["split_by_title"]
        self._model_path = os.path.join(
            paths_cfg["downloaded_models"], paths_cfg["models"]["jina_embeddings"]
        )
        self._index_dir = get_slg_index_dir(experiment)

        self._allowed = set(allowed_experts) if allowed_experts is not None else None
        self._model = None  # lazily loaded only when (re)building the cache

        self.expert_ids: List[str] = []
        self._matrix: np.ndarray = np.empty((0, self._embedding_dim), dtype="float32")
        self._load_or_build()

    # ------------------------------------------------------------------ build
    def _expert_files(self) -> List[Tuple[str, str]]:
        """(expert_id, data_path) for every split file kept by the allow-list."""
        files = sorted(f for f in os.listdir(self._split_dir) if f.endswith(".json"))
        pairs = []
        for file in files:
            expert_id = os.path.splitext(file)[0]
            if self._allowed is not None and expert_id not in self._allowed:
                continue
            pairs.append((expert_id, os.path.join(self._split_dir, file)))
        return pairs

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading Jina embedding model from %s", self._model_path)
            self._model = SentenceTransformer(self._model_path, trust_remote_code=True)
        return self._model

    def _embed_answers(self, data_path: str) -> np.ndarray:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = [str(e.get("answer", "")).strip() for e in data if e.get("answer")]
        texts = list(dict.fromkeys(t for t in raw if t))  # dedup, keep order
        if not texts:
            raise ValueError(f"No training answers found in {data_path}")

        model = self._load_model()
        vectors: List[np.ndarray] = []
        for i in range(0, len(texts), self._embedding_batch_size):
            batch = texts[i : i + self._embedding_batch_size]
            vecs = model.encode(
                batch, batch_size=len(batch), convert_to_numpy=True, show_progress_bar=False
            )
            vectors.append(np.asarray(vecs, dtype="float32"))
        mean_vec = np.mean(np.concatenate(vectors, axis=0), axis=0).astype("float32")
        if mean_vec.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Unexpected embedding dimension {mean_vec.shape[0]} "
                f"(expected {self._embedding_dim})."
            )
        return mean_vec

    def _build(self) -> None:
        pairs = self._expert_files()
        if not pairs:
            raise ValueError(
                f"No routable expert splits found in {self._split_dir} "
                "(check split_by_title and the adapter/description allow-list)."
            )
        ids: List[str] = []
        rows: List[np.ndarray] = []
        for expert_id, data_path in pairs:
            logger.info("Embedding expert '%s' for cosine routing.", expert_id)
            ids.append(expert_id)
            rows.append(self._embed_answers(data_path))

        self.expert_ids = ids
        self._matrix = _l2_normalize(np.stack(rows, axis=0).astype("float32"))
        self._save()

    # ------------------------------------------------------------------ cache
    def _cache_paths(self) -> Tuple[str, str]:
        return (
            os.path.join(self._index_dir, "expert_embeddings.npy"),
            os.path.join(self._index_dir, "expert_ids.json"),
        )

    def _save(self) -> None:
        ensure_dir(self._index_dir)
        emb_path, ids_path = self._cache_paths()
        np.save(emb_path, self._matrix)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.expert_ids, f, indent=2)
        logger.info("Cached %d expert routing embeddings to %s", len(self.expert_ids), self._index_dir)

    def _load_or_build(self) -> None:
        emb_path, ids_path = self._cache_paths()
        if os.path.isfile(emb_path) and os.path.isfile(ids_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            expected = {eid for eid, _ in self._expert_files()}
            if set(cached_ids) == expected and expected:
                self.expert_ids = cached_ids
                self._matrix = np.load(emb_path).astype("float32")
                logger.info("Loaded cached expert routing embeddings (%d experts).", len(cached_ids))
                return
            logger.info("Expert set changed since cache was built; re-embedding.")
        self._build()

    # ------------------------------------------------------------------ query
    def embed_query(self, text: str) -> np.ndarray:
        """Return the L2-normalized embedding of a single query string."""
        model = self._load_model()
        vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        vec = np.asarray(vec, dtype="float32")[0]
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-12)

    def shortlist(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        adjustments: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Rank experts by cosine similarity plus signed competence adjustments.

        ``adjustments`` are the online-competence deltas (see
        :mod:`inference.slg.competence`): positive boosts an expert that has
        proven reliable on similar questions, negative demotes one that has
        failed. They are added to the raw cosine score before ranking.
        """
        if self._matrix.shape[0] == 0:
            return []
        sims = self._matrix @ query_embedding  # both L2-normalized -> cosine
        scores: Dict[str, float] = {
            eid: float(sims[i]) for i, eid in enumerate(self.expert_ids)
        }
        if adjustments:
            for eid, delta in adjustments.items():
                if eid in scores:
                    scores[eid] += delta
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[: max(top_k, 0)]
