import json
import os
from typing import List, Tuple

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG
from logging_config import logger
from utils.path_utils import ensure_dir


def _slg_expert_dir_name(adapter_stem: str) -> str:
    """Directory name under experiments/<exp>/<slg_dir>; matches finetuned_<expert_stem> on disk."""
    if adapter_stem.startswith("finetuned_"):
        return adapter_stem
    return f"finetuned_{adapter_stem}"


def compute_chunk_embedding(
    data_path: str,
    model: SentenceTransformer,
    embedding_dimension: int,
    embedding_batch_size: int,
) -> np.ndarray:
    with open(data_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    # Neighbor similarity should reflect topic/content overlap between experts.
    # Embed training answers (where the knowledge lives), then average to one vector per chunk.
    raw = [
        str(entry.get("answer", "")).strip()
        for entry in chunk_data
        if entry.get("answer")
    ]
    # Deduplicate (first occurrence wins) so repeated answers are not double-counted in the mean.
    texts = list(dict.fromkeys(t for t in raw if t))
    if not texts:
        raise ValueError(f"No training answers found in {data_path}")

    vectors: List[np.ndarray] = []
    for i in range(0, len(texts), embedding_batch_size):
        batch = texts[i : i + embedding_batch_size]
        batch_vectors = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors.append(np.asarray(batch_vectors, dtype="float32"))

    all_vecs = np.concatenate(vectors, axis=0)
    mean_vec = np.mean(all_vecs, axis=0).astype("float32")
    if mean_vec.shape[0] != embedding_dimension:
        raise ValueError(
            f"Unexpected embedding dimension {mean_vec.shape[0]} "
            f"(expected {embedding_dimension})."
        )
    return mean_vec


def collect_slg_chunk_embeddings() -> Tuple[List[str], List[np.ndarray]]:
    """
    split_by_title files -> expert_ids (finetuned_<stem>) + chunk_embeddings.
    (No finetune call — training is separate.)
    """
    paths_config = CONFIG["paths"]
    split_by_title_dir = paths_config["split_by_title"]

    embedding_model_path = os.path.join(
        paths_config["downloaded_models"],
        paths_config["models"]["jina_embeddings"],
    )
    model = SentenceTransformer(embedding_model_path, trust_remote_code=True)

    slg_formation = CONFIG["slg_formation"]
    embedding_dimension = slg_formation["embedding_dimension"]
    embedding_batch_size = slg_formation.get("batch_size", 100)

    split_by_title_files = sorted(
        [f for f in os.listdir(split_by_title_dir) if f.endswith(".json")]
    )

    expert_ids: List[str] = []
    chunk_embeddings: List[np.ndarray] = []

    # Train SLG experts per title
    for file in split_by_title_files:
        logger.info(f"Training SLG expert for: {file}")
        adapter_name = os.path.splitext(file)[0]
        data_path = os.path.join(split_by_title_dir, file)

        # Compute a fixed representation for this training chunk.
        # expert_ids must match on-disk adapter folder names (finetuned_<stem>).
        expert_ids.append(_slg_expert_dir_name(adapter_name))
        chunk_embeddings.append(
            compute_chunk_embedding(
                data_path,
                model,
                embedding_dimension,
                embedding_batch_size,
            )
        )

    return expert_ids, chunk_embeddings


def save_slg_embedding_artifacts(
    experiment: str,
    expert_ids: List[str],
    chunk_embeddings: List[np.ndarray],
) -> None:
    """
    Build index.json via hnswlib inner-product search on L2-normalized chunk embeddings.
    Additionally writes chunk_embeddings_raw.npy and expert_ids.json (for reuse without API calls).
    """
    paths_config = CONFIG["paths"]
    experiments_dir = paths_config["experiments"]
    slg_formation = CONFIG["slg_formation"]
    embedding_dimension = slg_formation["embedding_dimension"]
    neighbor_k = slg_formation["k_neighbours"]

    # Build SLG cosine-similarity index across chunk embeddings.
    # This is consumed by inference/slg.py to expand to up to k neighbors.
    logger.info("Building SLG similarity index...")
    slg_dir = os.path.join(experiments_dir, experiment, slg_formation["slg_dir"])
    ensure_dir(slg_dir)

    if not expert_ids or not chunk_embeddings:
        raise ValueError("No SLG experts were trained; cannot build similarity index.")

    embeddings_matrix = np.stack(chunk_embeddings, axis=0).astype("float32")

    # Extra persistence (not in original inline block): raw matrix + id list for offline reuse.
    np.save(os.path.join(slg_dir, "chunk_embeddings_raw.npy"), embeddings_matrix)
    with open(os.path.join(slg_dir, "expert_ids.json"), "w", encoding="utf-8") as f:
        json.dump(expert_ids, f, indent=2)

    # Cosine similarity = inner product on L2-normalized vectors.
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized = embeddings_matrix / np.clip(norms, 1e-12, None)

    n = len(expert_ids)
    search_k = min(neighbor_k + 1, n)

    if n <= 1:
        # Degenerate graph: only self-neighbors exist; match knn_query output shape.
        indices = np.tile(np.arange(n, dtype=np.int64), (search_k, 1)).T
    else:
        index = hnswlib.Index(space="ip", dim=embedding_dimension)
        m = min(32, max(2, n - 1))
        ef_construction = max(200, search_k * 5)
        index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        index.add_items(normalized, np.arange(n, dtype=np.int64))
        index.set_ef(max(64, search_k * 5))
        labels, _distances = index.knn_query(normalized, k=search_k)
        indices = labels.astype(np.int64, copy=False)

    index_entries = []
    for i, expert_id in enumerate(expert_ids):
        neighbor_ids: List[str] = []
        for j in indices[i]:
            candidate_id = expert_ids[int(j)]
            if candidate_id == expert_id:
                continue
            neighbor_ids.append(candidate_id)
            if len(neighbor_ids) >= neighbor_k:
                break

        index_entries.append(
            {
                "expert_id": expert_id,
                "chunk_embedding": normalized[i].astype("float32").tolist(),
                "adapter_path": os.path.join(slg_dir, expert_id),
                "top_k_neighbors": neighbor_ids,
            }
        )

    index_path = os.path.join(slg_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, indent=2)
    logger.info(f"Wrote SLG index to: {index_path}")


def run_slg_embeddings(experiment: str) -> None:
    logger.info("Running SLG embedding + index step (no finetuning)...")
    expert_ids, chunk_embeddings = collect_slg_chunk_embeddings()
    save_slg_embedding_artifacts(experiment, expert_ids, chunk_embeddings)
