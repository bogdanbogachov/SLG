import os
import json
from typing import List

import numpy as np
import faiss
from openai import OpenAI

from finetune.finetune import finetune
from config import CONFIG
from logging_config import logger
from utils.path_utils import ensure_dir


def run_training(experiment: str):
    """
    Run training for specified model components.

    Args:
        experiment: Experiment identifier
    """
    paths_config = CONFIG['paths']
    files_config = CONFIG['files']
    models_paths = paths_config['models']
    adapters_config = CONFIG['adapters']

    # Get training configuration from config
    training_config = CONFIG.get('training_components', {})

    train_slg_system = training_config.get('train_slg_system', False)
    train_3_2_1b = training_config.get('train_3_2_1b', False)
    train_3_1_8b = training_config.get('train_3_1_8b', False)

    experiments_dir = paths_config['experiments']
    split_by_title_dir = paths_config['split_by_title']
    downloaded_models_dir = paths_config['downloaded_models']

    os.makedirs(experiments_dir, exist_ok=True)

    # Finetune SLG experts per title and orchestrator
    if train_slg_system:
        logger.info("Training SLG system (experts + orchestrator)...")

        # Build per-expert chunk embeddings first, then build a similarity index later.
        # Embeddings are computed before fine-tuning each expert.
        client = OpenAI(api_key=CONFIG["open_ai_api_key"])
        models_config = CONFIG["models"]
        embedding_model = models_config["embedding_model"]

        slg_formation = CONFIG["slg_formation"]
        embedding_dimension = slg_formation["embedding_dimension"]
        neighbor_k = slg_formation["k_neighbours"]
        embedding_batch_size = slg_formation.get("batch_size", 100)

        split_by_title_files = sorted(
            [f for f in os.listdir(split_by_title_dir) if f.endswith(".json")]
        )

        expert_ids: List[str] = []
        chunk_embeddings: List[np.ndarray] = []

        def compute_chunk_embedding(data_path: str) -> np.ndarray:
            with open(data_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)

            texts = [
                entry.get("answer", "")
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

        # Train SLG experts per title
        for file in split_by_title_files:
            logger.info(f"Training SLG expert for: {file}")
            adapter_name = os.path.splitext(file)[0]
            data_path = os.path.join(split_by_title_dir, file)

            # Compute a fixed representation for this training chunk.
            expert_ids.append(adapter_name)
            chunk_embeddings.append(compute_chunk_embedding(data_path))

            finetune(
                model_to_tune=os.path.join(
                    downloaded_models_dir, models_paths["3_2_1b"]
                ),
                adapter_name=adapter_name,
                data=data_path,
                experiment_number=experiment,
                slg=True,
            )

        # Train orchestrator
        logger.info("Training orchestrator...")
        finetune(
            model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
            adapter_name=adapters_config['orchestrator_3_2_1b'],
            data=files_config['qa_train'],
            experiment_number=experiment,
            orchestrator=True
        )

        # Build SLG cosine-similarity index across chunk embeddings.
        # This is consumed by inference/slg.py to expand to up to k neighbors.
        logger.info("Building SLG similarity index...")
        slg_dir =  slg_formation["slg_dir"]
        slg_dir = os.path.join(experiments_dir, experiment, slg_dir)
        ensure_dir(slg_dir)

        if not expert_ids or not chunk_embeddings:
            raise ValueError("Cannot build similarity index.")

        embeddings_matrix = np.stack(chunk_embeddings, axis=0).astype("float32")

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

    else:
        logger.info("Skipping SLG system training")

    # Baseline 3_2_1b
    if train_3_2_1b:
        logger.info("Training baseline model: 3_2_1b")
        finetune(
            model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
            adapter_name=adapters_config['finetuned_3_2_1b'],
            data=files_config['qa_train'],
            experiment_number=experiment
        )
    else:
        logger.info("Skipping baseline 3_2_1b training")

    # Baseline 3_1_8b
    if train_3_1_8b:
        logger.info("Training baseline model: 3_1_8b")
        finetune(
            model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_1_8b']),
            adapter_name=adapters_config['finetuned_3_1_8b'],
            data=files_config['qa_train'],
            experiment_number=experiment
        )
    else:
        logger.info("Skipping baseline 3_1_8b training")
