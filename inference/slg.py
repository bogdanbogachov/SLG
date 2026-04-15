"""Small Language Graph (SLG) for multi-expert question answering.

Pipeline:
1) Embed the question (Jina); pick the main expert by cosine similarity to chunk embeddings in index.json.
2) Main expert generates an answer + confidence (avg token log-prob -> exp -> [0, 1]).
3) Based on confidence, we invoke 1..k neighboring experts (neighbors by cosine similarity over
   chunk embeddings computed at training time).
4) The final answer is the candidate with the highest confidence score.
"""

import functools
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import torch
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

from config import CONFIG
from logging_config import logger
from utils.model_loader import cleanup_model_memory, load_model_with_adapter
from utils.path_utils import (
    ensure_dir,
    get_slg_index_dir,
    get_slg_path,
    validate_dir_exists,
    validate_slg_embedding_artifacts,
)
from utils.prompt_utils import apply_chat_template, create_user_message


class SmallLanguageGraph:
    def __init__(self, experts_location: str, experiment: str):
        self.experts_location = experts_location
        self.experiment = experiment

        paths_config = CONFIG["paths"]
        self.experiments_dir = paths_config["experiments"]
        self.slg_path = get_slg_path(self.experts_location, self.experiments_dir)
        self.index_dir = get_slg_index_dir(self.experiments_dir)
        validate_slg_embedding_artifacts(self.index_dir)
        validate_dir_exists(
            self.slg_path,
            error_message=(
                f"SLG expert adapters directory not found: {self.slg_path}. "
                "Train SLG experts for this experiment (training_components.train_slg_system) "
                f"so adapters exist under experiments/<experiment>/{CONFIG['slg_formation']['slg_dir']}/."
            ),
        )
        self.index_path = os.path.join(self.index_dir, "index.json")

        self.slg_index = self._load_slg_index()
        self.slg_neighbors_by_expert: Dict[str, List[str]] = self.slg_index[
            "neighbors_by_expert"
        ]
        self.neighbor_k: int = int(self.slg_index["neighbor_k"])
        self.slg_embeddings_by_expert: Dict[str, np.ndarray] = self.slg_index[
            "embeddings_by_expert"
        ]

        self.expert_nodes: List[str] = self._discover_expert_nodes()

        paths_cfg = CONFIG["paths"]
        jina_path = os.path.join(
            paths_cfg["downloaded_models"],
            paths_cfg["models"]["jina_embeddings"],
        )
        self._embedding_model = SentenceTransformer(jina_path, trust_remote_code=True)

    def _discover_expert_nodes(self) -> List[str]:
        if not os.path.isdir(self.slg_path):
            raise FileNotFoundError(
                f"SLG experts directory not found: {self.slg_path}. "
                f"Run training to create experiments/<exp>/slg."
            )

        # Expert nodes are adapter directories; ignore JSON files like index.json.
        return sorted(
            [
                name
                for name in os.listdir(self.slg_path)
                if not name.endswith(".json")
                and os.path.isdir(os.path.join(self.slg_path, name))
            ]
        )

    def _load_slg_index(self) -> Dict[str, Any]:
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(
                f"SLG similarity index not found: {self.index_path}. "
                "Run commands.slg_embeddings.run_slg_embeddings (writes under experiments/<slg_index>/)."
            )

        with open(self.index_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        neighbors_by_expert: Dict[str, List[str]] = {}
        embeddings_by_expert: Dict[str, np.ndarray] = {}
        neighbor_list_lengths: List[int] = []
        for e in entries:
            expert_id = e["expert_id"]
            top_neighbors = e.get("top_k_neighbors", [])
            neighbors_by_expert[expert_id] = list(top_neighbors)
            neighbor_list_lengths.append(len(top_neighbors))
            vec = np.asarray(e["chunk_embedding"], dtype=np.float32).reshape(-1)
            embeddings_by_expert[expert_id] = vec

        # Upper bound for confidence→how-many-neighbors scheduling (experts may have 0..cap links).
        neighbor_k = max(neighbor_list_lengths) if neighbor_list_lengths else 0

        return {
            "neighbors_by_expert": neighbors_by_expert,
            "neighbor_k": neighbor_k,
            "embeddings_by_expert": embeddings_by_expert,
        }

    def _route_question_by_embedding(
        self, question: str, candidate_experts: List[str]
    ) -> str:
        """Pick the expert whose index chunk embedding has highest cosine similarity to the question."""
        emb_map = self.slg_embeddings_by_expert
        expert_set = set(self.expert_nodes)
        candidates = [
            e for e in candidate_experts if e in emb_map and e in expert_set
        ]
        if not candidates:
            candidates = [e for e in self.expert_nodes if e in emb_map]
        if not candidates:
            raise RuntimeError(
                "No experts with chunk embeddings in index.json match on-disk SLG adapters. "
                "Run commands.slg_embeddings.run_slg_embeddings for this experiment."
            )

        q = self._embedding_model.encode(
            [question],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        qn = float(np.linalg.norm(q))
        if qn < 1e-12:
            return candidates[0]
        q = q / qn

        best_e = candidates[0]
        best_s = -1.0
        for e in candidates:
            sim = float(np.dot(q, emb_map[e]))
            if sim > best_s:
                best_s = sim
                best_e = e
        return best_e

    def _task_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route the question to the main expert via embedding similarity to index.json vectors."""
        question = state["question"]
        on_disk = set(self.expert_nodes)
        with_emb = set(self.slg_embeddings_by_expert.keys())
        experts_list_of_strings = sorted(on_disk & with_emb)
        if not experts_list_of_strings:
            experts_list_of_strings = list(self.expert_nodes)

        state["selected_expert"] = self._route_question_by_embedding(
            question, experts_list_of_strings
        )
        return state

    @staticmethod
    def _extract_assistant_text(decoded: str) -> str:
        if "assistant" in decoded:
            return decoded.split("assistant")[-1].strip()
        return decoded.strip()

    def _tuned_generate_with_confidence(
        self, prompt: str, adapter: str
    ) -> Dict[str, Any]:
        """Generate answer with a tuned expert adapter and compute confidence."""
        messages = [create_user_message(prompt)]

        paths_config = CONFIG["paths"]
        models_paths = paths_config["models"]
        base_model_path = os.path.join(
            paths_config["downloaded_models"], models_paths["3_2_1b"]
        )

        finetuned_model, tokenizer = load_model_with_adapter(
            base_model_path=base_model_path,
            adapter_path=adapter,
            resize_token_embeddings=True,  # SLG experts need this
        )

        try:
            formatted_prompt = apply_chat_template(
                messages, tokenizer, add_generation_prompt=True
            )
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
            ).to("cuda")

            generation_config = CONFIG["generation"]
            eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                num_return_sequences=1,
                temperature=generation_config["temperature"],
                eos_token_id=eos_id,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            answer_text = self._extract_assistant_text(decoded)

            prompt_len = int(inputs["input_ids"].shape[1])
            gen_scores = outputs.scores
            if not gen_scores:
                return {"answer": answer_text, "confidence": 0.0}

            seq0 = outputs.sequences[0]
            logprobs: List[float] = []
            for i, step_scores in enumerate(gen_scores):
                token_id = int(seq0[prompt_len + i].item())
                logits = step_scores[0]  # [vocab]
                token_logprob = torch.log_softmax(logits, dim=-1)[token_id].item()
                logprobs.append(token_logprob)

            avg_logprob = float(sum(logprobs) / len(logprobs))
            confidence = math.exp(avg_logprob)
            confidence = max(0.0, min(1.0, confidence))

            return {"answer": answer_text, "confidence": confidence}
        finally:
            cleanup_model_memory(finetuned_model, tokenizer)

    def _expert_node_builder(
        self, state: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """Run a tuned expert adapter and append its answer to the graph state."""
        question = state["question"]
        expert_adapter_path = os.path.join(self.slg_path, model)

        result = self._tuned_generate_with_confidence(question, expert_adapter_path)
        logger.info(f"Expert '{model}' confidence: {float(result['confidence']):.4f}")

        state["answers"].append(
            {
                "expert": model,
                "confidence": float(result["confidence"]),
                "answer": result["answer"],
            }
        )
        state["visited_experts"].append(model)
        state["last_expert"] = model

        if state.get("selected_expert") == model and state.get("phase") == "main":
            state["main_confidence"] = float(result["confidence"])

        return state

    def _confidence_to_neighbor_count(self, confidence: float, k: int) -> int:
        """Map confidence in [0, 1] to how many neighbors to invoke."""
        if k <= 0:
            return 0
        if confidence >= 0.85:
            return 0
        if confidence >= 0.75:
            return min(1, k)
        if confidence >= 0.65:
            return min(2, k)
        return k

    def _confidence_router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide how many neighboring experts to invoke and set pending neighbors."""
        selected_expert = state.get("selected_expert")
        if not selected_expert:
            state["pending_neighbors"] = []
            state["phase"] = "neighbors"
            return state

        main_conf = state.get("main_confidence")
        if main_conf is None:
            for a in state.get("answers", []):
                if a.get("expert") == selected_expert:
                    main_conf = a.get("confidence")
                    break

        main_conf = float(main_conf or 0.0)
        m = self._confidence_to_neighbor_count(main_conf, self.neighbor_k)

        neighbors = self.slg_neighbors_by_expert.get(selected_expert, [])
        visited = set(state.get("visited_experts", []))
        pending: List[str] = []
        for nid in neighbors[:m]:
            if nid == selected_expert:
                continue
            if nid in visited:
                continue
            if nid not in set(self.expert_nodes):
                continue
            pending.append(nid)

        state["pending_neighbors"] = pending
        state["phase"] = "neighbors"
        return state

    def _route_to_main_expert(self, state: Dict[str, Any]) -> str:
        selected_expert = state.get("selected_expert")
        if selected_expert in set(self.expert_nodes):
            return selected_expert
        return END

    def _route_from_confidence_router(self, state: Dict[str, Any]) -> str:
        pending = state.get("pending_neighbors", [])
        if pending:
            next_expert = pending.pop(0)
            state["pending_neighbors"] = pending
            return next_expert
        return "aggregator"

    def _route_after_expert(self, state: Dict[str, Any]) -> str:
        if state.get("phase") == "main":
            return "confidence_router"

        pending = state.get("pending_neighbors", [])
        if pending:
            next_expert = pending.pop(0)
            state["pending_neighbors"] = pending
            return next_expert
        return "aggregator"

    def _aggregator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Choose the expert answer with the highest confidence as the final answer."""
        candidates = state.get("answers", [])
        if not candidates:
            state["final_answer"] = ""
            return state

        best = max(
            candidates,
            key=lambda c: float(c.get("confidence") or 0.0),
        )
        state["final_answer"] = str(best.get("answer") or "")
        return state

    def _build_graph(self):
        logger.info("Building SLG graph.")
        graph_builder = StateGraph(dict)

        graph_builder.add_node("orchestrator", self._task_analysis_node)
        graph_builder.add_node("confidence_router", self._confidence_router_node)
        graph_builder.add_node("aggregator", self._aggregator_node)

        for expert in self.expert_nodes:
            graph_builder.add_node(
                expert,
                functools.partial(
                    self._expert_node_builder,
                    model=expert,
                ),
            )

        graph_builder.add_edge(START, "orchestrator")
        graph_builder.add_conditional_edges("orchestrator", self._route_to_main_expert)

        for expert in self.expert_nodes:
            graph_builder.add_conditional_edges(expert, self._route_after_expert)

        graph_builder.add_conditional_edges(
            "confidence_router", self._route_from_confidence_router
        )
        graph_builder.add_edge("aggregator", END)

        return graph_builder.compile()

    def ask_slg(self, file: str) -> None:
        """Run the SLG graph for all questions in a file."""
        from utils.path_utils import validate_file_exists

        validate_file_exists(file)

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paths_config = CONFIG["paths"]
        output_dir = os.path.join(paths_config["answers"], self.experiment)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, "slg.json")

        # Load existing progress if available
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                answers_list: List[Dict[str, Any]] = json.load(f)
            start_index = len(answers_list)
            logger.info(f"Resuming SLG inference from index {start_index}/{len(data)}.")
        else:
            answers_list = []
            start_index = 0
            logger.info("Starting fresh SLG inference run.")

        graph = self._build_graph()

        for i, item in enumerate(data[start_index:], start=start_index):
            logger.info(f"Answering {i + 1}/{len(data)} questions.")
            logger.info(f"Inference of the title: {item['title']}")
            initial_state: Dict[str, Any] = {
                "question": item["question"],
                "phase": "main",
                "selected_expert": None,
                "visited_experts": [],
                "answers": [],
                "pending_neighbors": [],
                "main_confidence": None,
                "last_expert": None,
            }
            result = graph.invoke(initial_state)

            answers_list.append(
                {
                    "chapter": item["chapter"],
                    "title": item["title"],
                    "question": item["question"],
                    "answer": result.get("final_answer"),
                }
            )

            # Save progress incrementally so we can resume after interruptions.
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(answers_list, f, indent=4)
            logger.info(40 * "-")

        return None
