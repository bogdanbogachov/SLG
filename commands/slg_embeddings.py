import json
import os
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG
from logging_config import logger
from utils.path_utils import ensure_dir, get_slg_index_dir, slg_expert_id_from_filename


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
        adapter_name = slg_expert_id_from_filename(file)
        data_path = os.path.join(split_by_title_dir, file)

        # Compute a fixed representation for this training chunk.
        # expert_ids must match on-disk adapter folder names (finetuned_<stem>).
        expert_ids.append(adapter_name)
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
    Build index.json from pairwise cosine similarity on L2-normalized chunk embeddings.

    Neighbors: experts j != i with similarity >= neighbor_similarity_threshold, sorted by
    similarity descending, keep at most k_neighbours. If none pass the threshold, [].
    Also writes chunk_embeddings_raw.npy and expert_ids.json.
    """
    paths_config = CONFIG["paths"]
    experiments_dir = paths_config["experiments"]
    slg_formation = CONFIG["slg_formation"]
    max_neighbors = int(slg_formation["k_neighbours"])
    threshold = float(slg_formation.get("neighbor_similarity_threshold", 0.95))

    # Per-experiment index directory; expert adapters stay under experiments/<exp>/<slg_dir>/
    index_dir = get_slg_index_dir(experiment, experiments_dir)
    experts_slg_dir = os.path.join(experiments_dir, experiment, slg_formation["slg_dir"])

    logger.info(
        "Building SLG similarity index (threshold=%s, max_neighbors=%s)...",
        threshold,
        max_neighbors,
    )
    ensure_dir(index_dir)

    if not expert_ids or not chunk_embeddings:
        raise ValueError("No SLG experts were trained; cannot build similarity index.")

    embeddings_matrix = np.stack(chunk_embeddings, axis=0).astype("float32")

    # Extra persistence (not in original inline block): raw matrix + id list for offline reuse.
    np.save(os.path.join(index_dir, "chunk_embeddings_raw.npy"), embeddings_matrix)
    with open(os.path.join(index_dir, "expert_ids.json"), "w", encoding="utf-8") as f:
        json.dump(expert_ids, f, indent=2)

    # Cosine similarity = inner product on L2-normalized vectors.
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized = embeddings_matrix / np.clip(norms, 1e-12, None)

    n = len(expert_ids)
    if n > 1:
        sim_matrix = normalized @ normalized.T

    index_entries = []
    for i, expert_id in enumerate(expert_ids):
        neighbor_ids: List[str] = []
        if n > 1:
            sims = sim_matrix[i]
            candidates = [
                j for j in range(n) if j != i and float(sims[j]) >= threshold
            ]
            candidates.sort(key=lambda j: float(sims[j]), reverse=True)
            for j in candidates[:max_neighbors]:
                neighbor_ids.append(expert_ids[j])

        index_entries.append(
            {
                "expert_id": expert_id,
                "chunk_embedding": normalized[i].astype("float32").tolist(),
                "adapter_path": os.path.join(experts_slg_dir, expert_id),
                "top_k_neighbors": neighbor_ids,
            }
        )

    index_path = os.path.join(index_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, indent=2)
    logger.info(f"Wrote SLG index to: {index_path}")


def run_slg_embeddings(experiment: str) -> None:
    paths_config = CONFIG["paths"]
    experiments_dir = paths_config["experiments"]
    index_dir = get_slg_index_dir(experiment, experiments_dir)
    index_path = os.path.join(index_dir, "index.json")
    if os.path.isfile(index_path):
        logger.info(
            "SLG index already exists at %s; skipping embedding + index rebuild. "
            "Delete this file to force a rebuild.",
            index_path,
        )
        return

    logger.info("Running SLG embedding + index step (no finetuning)...")
    expert_ids, chunk_embeddings = collect_slg_chunk_embeddings()
    save_slg_embedding_artifacts(experiment, expert_ids, chunk_embeddings)
