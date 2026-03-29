import json
import os
import time
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI

from config import CONFIG
from logging_config import logger
from utils.path_utils import ensure_dir


def compute_chunk_embedding(
    data_path: str,
    client: OpenAI,
    embedding_model: str,
    embedding_dimension: int,
    embedding_batch_size: int,
) -> np.ndarray:
    with open(data_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    # Neighbor similarity should reflect topic/content overlap between experts.
    # Embed training answers (where the knowledge lives), then average to one vector per chunk.
    texts = [
        str(entry.get("answer", "")).strip()
        for entry in chunk_data
        if entry.get("answer")
    ]
    if not texts:
        raise ValueError(f"No training answers found in {data_path}")

    vectors: List[np.ndarray] = []
    for i in range(0, len(texts), embedding_batch_size):
        batch = texts[i : i + embedding_batch_size]
        response = client.embeddings.create(
            model=embedding_model,
            input=batch,
        )
        time.sleep(0.5)
        batch_vectors = np.array(
            [item.embedding for item in response.data], dtype="float32"
        )
        vectors.append(batch_vectors)

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
    Same loop as train.run_training SLG branch: split_by_title files -> expert_ids + chunk_embeddings.
    (No finetune call — that is the only difference from train.py.)
    """
    paths_config = CONFIG["paths"]
    split_by_title_dir = paths_config["split_by_title"]

    client = OpenAI(api_key=CONFIG["open_ai_api_key"])
    models_config = CONFIG["models"]
    embedding_model = models_config["embedding_model"]

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
        expert_ids.append(adapter_name)
        chunk_embeddings.append(
            compute_chunk_embedding(
                data_path,
                client,
                embedding_model,
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
    Same FAISS + index.json logic as the end of the SLG branch in train.py.
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

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(normalized)

    # Query each embedding against the index (k neighbors + 1 to account for self).
    search_k = min(neighbor_k + 1, len(expert_ids))
    _, indices = index.search(normalized, search_k)

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
