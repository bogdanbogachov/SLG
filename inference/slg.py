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
import time
from typing import Any, Dict, List

import numpy as np
import torch
from langgraph.graph import END, START, StateGraph
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONFIG
from finetune.router import ROUTER_METADATA_FILE
from logging_config import logger
from utils.model_loader import cleanup_model_memory, load_model_with_adapter
from utils.path_utils import (
    ensure_dir,
    get_slg_index_dir,
    get_slg_path,
    slg_expert_id_from_filename,
    validate_dir_exists,
    validate_file_exists,
    validate_slg_embedding_artifacts,
)
from utils.prompt_utils import apply_chat_template, create_user_message


ROUTER_COSINE = "cosine"
ROUTER_FINETUNED = "finetuned"
SUPPORTED_ROUTER_METHODS = {ROUTER_COSINE, ROUTER_FINETUNED}


class SmallLanguageGraph:
    def __init__(
        self,
        experts_location: str,
        experiment: str,
        router_method: str = None,
    ):
        self.experts_location = experts_location
        self.experiment = experiment
        self.router_method = router_method or CONFIG.get("routing", {}).get(
            "method", ROUTER_COSINE
        )
        if self.router_method not in SUPPORTED_ROUTER_METHODS:
            supported = ", ".join(sorted(SUPPORTED_ROUTER_METHODS))
            raise ValueError(
                f"Unsupported SLG router method: {self.router_method!r}. "
                f"Expected one of: {supported}."
            )

        paths_config = CONFIG["paths"]
        self.experiments_dir = paths_config["experiments"]
        self.slg_path = get_slg_path(self.experts_location, self.experiments_dir)
        validate_dir_exists(
            self.slg_path,
            error_message=(
                f"SLG expert adapters directory not found: {self.slg_path}. "
                "Train SLG experts for this experiment (training_components.train_slg_system) "
                f"so adapters exist under experiments/<experiment>/{CONFIG['slg_formation']['slg_dir']}/."
            ),
        )
        self._compiled_graph = None

        self.expert_nodes: List[str] = self._discover_expert_nodes()
        # LangGraph reserves some characters (notably ":") in node names, while
        # expert adapter directories are derived from titles and may contain them.
        # Keep adapter/expert ids unchanged for paths and index lookup, but use
        # deterministic safe ids inside the graph.
        self.graph_node_by_expert: Dict[str, str] = {
            expert: f"expert_{i:05d}" for i, expert in enumerate(self.expert_nodes)
        }

        self.index_dir = None
        self.index_path = None
        self.slg_index: Dict[str, Any] = {}
        self.slg_neighbors_by_expert: Dict[str, List[str]] = {}
        self.neighbor_k = 0
        self.slg_embeddings_by_expert: Dict[str, np.ndarray] = {}
        self._embedding_model = None

        self.router_adapter_path = None
        self.router_metadata_path = None
        self.router_metadata: Dict[str, Any] = {}

        if self.router_method == ROUTER_COSINE:
            self._init_cosine_router(paths_config)
        else:
            self._init_finetuned_router(paths_config)

    def _init_cosine_router(self, paths_config: Dict[str, Any]) -> None:
        self.index_dir = get_slg_index_dir(self.experts_location, self.experiments_dir)
        validate_slg_embedding_artifacts(self.index_dir)
        self.index_path = os.path.join(self.index_dir, "index.json")

        self.slg_index = self._load_slg_index()
        self.slg_neighbors_by_expert = self.slg_index["neighbors_by_expert"]
        self.neighbor_k = int(self.slg_index["neighbor_k"])
        self.slg_embeddings_by_expert = self.slg_index["embeddings_by_expert"]

        jina_path = os.path.join(
            paths_config["downloaded_models"],
            paths_config["models"]["jina_embeddings"],
        )
        self._embedding_model = SentenceTransformer(jina_path, trust_remote_code=True)

    def _init_finetuned_router(self, paths_config: Dict[str, Any]) -> None:
        adapter_name = CONFIG["adapters"]["slg_router_3_2_1b"]
        self.router_adapter_path = os.path.join(
            paths_config["experiments"],
            self.experiment,
            adapter_name,
        )
        self.router_metadata_path = os.path.join(
            self.router_adapter_path,
            ROUTER_METADATA_FILE,
        )
        validate_dir_exists(
            self.router_adapter_path,
            error_message=(
                f"Fine-tuned SLG router adapter not found: {self.router_adapter_path}. "
                "Train it with training_components.train_slg_router before using "
                "--router finetuned."
            ),
        )
        validate_file_exists(
            self.router_metadata_path,
            error_message=(
                f"Fine-tuned SLG router metadata not found: {self.router_metadata_path}. "
                "Re-train the router so label mappings are saved."
            ),
        )
        with open(self.router_metadata_path, "r", encoding="utf-8") as f:
            self.router_metadata = json.load(f)

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
                "Run commands.slg_embeddings.run_slg_embeddings "
                "(writes under experiments/<experiment>/<slg_index>/)."
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

    def _rank_question_by_embedding(
        self, question: str, candidate_experts: List[str]
    ) -> List[Dict[str, Any]]:
        """Rank experts by cosine similarity to the question embedding."""
        if self._embedding_model is None:
            raise RuntimeError("Cosine router is not initialized.")
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
            return [
                {
                    "rank": i + 1,
                    "expert": expert,
                    "score": None,
                    "score_type": "cosine_similarity",
                }
                for i, expert in enumerate(candidates)
            ]
        q = q / qn

        ranked = [
            {
                "expert": expert,
                "score": float(np.dot(q, emb_map[expert])),
                "score_type": "cosine_similarity",
            }
            for expert in candidates
        ]
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        for i, item in enumerate(ranked):
            item["rank"] = i + 1
        return ranked

    def _route_question_by_embedding(
        self, question: str, candidate_experts: List[str]
    ) -> str:
        """Pick the highest-cosine expert."""
        return self._rank_question_by_embedding(question, candidate_experts)[0]["expert"]

    def _set_classifier_pad_token(
        self,
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

    def _rank_question_by_finetuned_router(self, question: str) -> List[Dict[str, Any]]:
        """Rank experts using the fine-tuned question -> expert classifier."""
        id2label_raw = self.router_metadata.get("id2label")
        if not isinstance(id2label_raw, dict) or not id2label_raw:
            raise RuntimeError(
                f"Invalid SLG router metadata in {self.router_metadata_path}: "
                "missing id2label mapping."
            )
        id2label = {int(idx): str(label) for idx, label in id2label_raw.items()}
        label2id = {label: idx for idx, label in id2label.items()}

        paths_config = CONFIG["paths"]
        base_model_path = os.path.join(
            paths_config["downloaded_models"],
            paths_config["models"]["3_2_1b"],
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.router_adapter_path,
            trust_remote_code=True,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        ).to(torch.device("cuda"))
        self._set_classifier_pad_token(tokenizer, base_model)
        router_model = None

        try:
            router_model = PeftModel.from_pretrained(
                base_model,
                self.router_adapter_path,
            )
            router_model = router_model.to(torch.device("cuda"))
            router_model.eval()
            inputs = tokenizer(
                question,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=CONFIG["data"]["max_length"],
            ).to("cuda")
            with torch.no_grad():
                outputs = router_model(**inputs)
            logits = outputs.logits[0].detach().float().cpu()
            probs = torch.softmax(logits, dim=-1)
            ranked: List[Dict[str, Any]] = []
            for label_id in torch.argsort(probs, descending=True).tolist():
                expert = id2label.get(int(label_id))
                if not expert:
                    continue
                ranked.append(
                    {
                        "rank": len(ranked) + 1,
                        "expert": expert,
                        "probability": float(probs[label_id].item()),
                        "logit": float(logits[label_id].item()),
                        "score": float(probs[label_id].item()),
                        "score_type": "classification_probability",
                    }
                )
            if not ranked:
                raise RuntimeError("Fine-tuned router produced no ranked candidates.")
            selected = ranked[0]["expert"]
            if selected not in set(self.expert_nodes):
                raise RuntimeError(
                    f"Fine-tuned router selected expert {selected!r}, but no matching "
                    f"adapter exists under {self.slg_path}."
                )
            return ranked
        finally:
            cleanup_model_memory(router_model or base_model, tokenizer)

    def _route_question_by_finetuned_router(self, question: str) -> str:
        """Pick the highest-probability fine-tuned router expert."""
        return self._rank_question_by_finetuned_router(question)[0]["expert"]

    def _task_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route the question to the main expert."""
        question = state["question"]
        start = time.perf_counter()

        if self.router_method == ROUTER_FINETUNED:
            ranked = self._rank_question_by_finetuned_router(question)
            state["selected_expert"] = ranked[0]["expert"]
            state["routing_candidates"] = ranked
            state["routing_latency_seconds"] = time.perf_counter() - start
            return state

        on_disk = set(self.expert_nodes)
        with_emb = set(self.slg_embeddings_by_expert.keys())
        experts_list_of_strings = sorted(on_disk & with_emb)
        if not experts_list_of_strings:
            experts_list_of_strings = list(self.expert_nodes)

        ranked = self._rank_question_by_embedding(question, experts_list_of_strings)
        state["selected_expert"] = ranked[0]["expert"]
        state["routing_candidates"] = ranked
        state["routing_latency_seconds"] = time.perf_counter() - start
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
            return self.graph_node_by_expert[selected_expert]
        return END

    def _route_from_confidence_router(self, state: Dict[str, Any]) -> str:
        pending = state.get("pending_neighbors", [])
        if pending:
            next_expert = pending.pop(0)
            state["pending_neighbors"] = pending
            return self.graph_node_by_expert[next_expert]
        return "aggregator"

    def _route_after_expert(self, state: Dict[str, Any]) -> str:
        if state.get("phase") == "main":
            if self.router_method == ROUTER_FINETUNED:
                return "aggregator"
            return "confidence_router"

        pending = state.get("pending_neighbors", [])
        if pending:
            next_expert = pending.pop(0)
            state["pending_neighbors"] = pending
            return self.graph_node_by_expert[next_expert]
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

    def _slg_output_basename(self) -> str:
        if self.router_method == ROUTER_COSINE:
            return "slg"
        return f"slg_{self.router_method}_router"

    def _other_router_report_path(self, report_path: str) -> str:
        other_basename = (
            "slg_finetuned_router"
            if self.router_method == ROUTER_COSINE
            else "slg"
        )
        return os.path.join(
            os.path.dirname(report_path),
            f"{other_basename}_routing_report.json",
        )

    @staticmethod
    def _expected_expert_id_from_title(title: str) -> str:
        split_title = (
            str(title)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\n", "_")
            .lower()
        )
        return slg_expert_id_from_filename(f"{split_title}.json")

    @staticmethod
    def _token_f1(reference: str, candidate: str) -> float:
        ref_tokens = str(reference or "").lower().split()
        cand_tokens = str(candidate or "").lower().split()
        if not ref_tokens and not cand_tokens:
            return 1.0
        if not ref_tokens or not cand_tokens:
            return 0.0
        ref_counts: Dict[str, int] = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = 0
        for token in cand_tokens:
            count = ref_counts.get(token, 0)
            if count:
                overlap += 1
                ref_counts[token] = count - 1
        if overlap == 0:
            return 0.0
        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _expected_rank(
        expected_expert: str,
        routing_candidates: List[Dict[str, Any]],
    ) -> int | None:
        for candidate in routing_candidates:
            if candidate.get("expert") == expected_expert:
                return int(candidate.get("rank") or 0) or None
        return None

    def _build_routing_record(
        self,
        index: int,
        item: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        expected_expert = self._expected_expert_id_from_title(item.get("title", ""))
        selected_expert = result.get("selected_expert")
        candidate_answers = result.get("answers", [])
        selected_candidate = next(
            (
                candidate
                for candidate in candidate_answers
                if candidate.get("expert") == selected_expert
            ),
            None,
        )
        final_answer = str(result.get("final_answer") or "")
        reference_answer = str(item.get("answer") or "")
        routing_candidates = result.get("routing_candidates", [])
        expected_rank = self._expected_rank(expected_expert, routing_candidates)
        top_router_candidates = routing_candidates[:10]
        selected_route = next(
            (
                candidate
                for candidate in routing_candidates
                if candidate.get("expert") == selected_expert
            ),
            None,
        )
        return {
            "index": index,
            "chapter": item.get("chapter"),
            "title": item.get("title"),
            "question": item.get("question"),
            "expected_expert": expected_expert,
            "expected_expert_exists": expected_expert in set(self.expert_nodes),
            "selected_expert": selected_expert,
            "routing_correct": bool(expected_expert and selected_expert)
            and expected_expert == selected_expert,
            "expected_rank": expected_rank,
            "top_router_candidates": top_router_candidates,
            "selected_router_score": (
                float(selected_route["score"])
                if selected_route and selected_route.get("score") is not None
                else None
            ),
            "selected_router_score_type": (
                selected_route.get("score_type") if selected_route else None
            ),
            "router_method": self.router_method,
            "visited_experts": result.get("visited_experts", []),
            "selected_confidence": (
                float(selected_candidate["confidence"])
                if selected_candidate and selected_candidate.get("confidence") is not None
                else None
            ),
            "candidate_answers": candidate_answers,
            "answer_quality": {
                "exact_match": int(final_answer.strip() == reference_answer.strip()),
                "token_f1": self._token_f1(reference_answer, final_answer),
                "reference_length_tokens": len(reference_answer.split()),
                "answer_length_tokens": len(final_answer.split()),
            },
            "latency_seconds": {
                "routing": result.get("routing_latency_seconds"),
                "total": result.get("total_latency_seconds"),
            },
        }

    @staticmethod
    def _routing_report_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(records)
        correct = sum(1 for record in records if record.get("routing_correct"))
        missing_expected = sum(
            1 for record in records if not record.get("expected_expert_exists")
        )
        by_expected: Dict[str, Dict[str, int]] = {}
        for record in records:
            expected = str(record.get("expected_expert") or "unknown")
            stats = by_expected.setdefault(expected, {"total": 0, "correct": 0})
            stats["total"] += 1
            if record.get("routing_correct"):
                stats["correct"] += 1

        for stats in by_expected.values():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] if stats["total"] else 0.0
            )

        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total else 0.0,
            "accuracy_ci_95_wilson": SmallLanguageGraph._wilson_ci(correct, total),
            "missing_expected_expert_count": missing_expected,
            "by_expected_expert": by_expected,
            "top_k_accuracy": SmallLanguageGraph._top_k_accuracy(records),
            "by_chapter": SmallLanguageGraph._accuracy_by_field(records, "chapter"),
            "confusion_matrix": SmallLanguageGraph._confusion_matrix(records),
            "answer_quality_by_routing_correctness": (
                SmallLanguageGraph._answer_quality_by_routing_correctness(records)
            ),
            "latency_seconds": SmallLanguageGraph._latency_summary(records),
            "error_examples": SmallLanguageGraph._error_examples(records),
        }

    @staticmethod
    def _wilson_ci(correct: int, total: int, z: float = 1.96) -> Dict[str, float | None]:
        if total <= 0:
            return {"low": None, "high": None}
        phat = correct / total
        denom = 1 + z * z / total
        center = (phat + z * z / (2 * total)) / denom
        margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
        return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}

    @staticmethod
    def _top_k_accuracy(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float | int]]:
        output: Dict[str, Dict[str, float | int]] = {}
        for k in (1, 3, 5, 10):
            eligible = [
                record
                for record in records
                if record.get("expected_rank") is not None
            ]
            correct = sum(
                1 for record in eligible if int(record["expected_rank"]) <= k
            )
            output[f"top_{k}"] = {
                "total": len(eligible),
                "correct": correct,
                "accuracy": correct / len(eligible) if eligible else 0.0,
            }
        return output

    @staticmethod
    def _accuracy_by_field(
        records: List[Dict[str, Any]],
        field: str,
    ) -> Dict[str, Dict[str, float | int]]:
        grouped: Dict[str, Dict[str, int]] = {}
        for record in records:
            key = str(record.get(field) or "unknown")
            stats = grouped.setdefault(key, {"total": 0, "correct": 0})
            stats["total"] += 1
            if record.get("routing_correct"):
                stats["correct"] += 1

        return {
            key: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            }
            for key, stats in sorted(grouped.items())
        }

    @staticmethod
    def _confusion_matrix(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {}
        for record in records:
            expected = str(record.get("expected_expert") or "unknown")
            selected = str(record.get("selected_expert") or "unknown")
            row = matrix.setdefault(expected, {})
            row[selected] = row.get(selected, 0) + 1
        return matrix

    @staticmethod
    def _mean(values: List[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _answer_quality_by_routing_correctness(
        records: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float | int | None]]:
        output: Dict[str, Dict[str, float | int | None]] = {}
        for label, desired in (("correct_routes", True), ("incorrect_routes", False)):
            bucket = [
                record
                for record in records
                if bool(record.get("routing_correct")) is desired
            ]
            exact = [
                float(record.get("answer_quality", {}).get("exact_match", 0))
                for record in bucket
            ]
            token_f1 = [
                float(record.get("answer_quality", {}).get("token_f1", 0.0))
                for record in bucket
            ]
            output[label] = {
                "total": len(bucket),
                "exact_match_rate": SmallLanguageGraph._mean(exact),
                "avg_token_f1": SmallLanguageGraph._mean(token_f1),
            }
        return output

    @staticmethod
    def _latency_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float | None]]:
        output: Dict[str, Dict[str, float | None]] = {}
        for key in ("routing", "total"):
            values = [
                float(record.get("latency_seconds", {}).get(key))
                for record in records
                if record.get("latency_seconds", {}).get(key) is not None
            ]
            if not values:
                output[key] = {"mean": None, "min": None, "max": None}
                continue
            output[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        return output

    @staticmethod
    def _error_examples(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        wrong = [record for record in records if not record.get("routing_correct")]
        correct = [record for record in records if record.get("routing_correct")]

        def compact(record: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "index": record.get("index"),
                "chapter": record.get("chapter"),
                "title": record.get("title"),
                "expected_expert": record.get("expected_expert"),
                "selected_expert": record.get("selected_expert"),
                "selected_router_score": record.get("selected_router_score"),
                "selected_confidence": record.get("selected_confidence"),
                "question": record.get("question"),
            }

        wrong.sort(
            key=lambda record: float(record.get("selected_router_score") or -1.0),
            reverse=True,
        )
        correct.sort(
            key=lambda record: float(record.get("selected_router_score") or 1e9)
        )
        return {
            "high_router_score_wrong_routes": [compact(record) for record in wrong[:10]],
            "low_router_score_correct_routes": [compact(record) for record in correct[:10]],
        }

    @staticmethod
    def _mcnemar_test(
        current_records: List[Dict[str, Any]],
        other_records: List[Dict[str, Any]],
    ) -> Dict[str, float | int | None]:
        other_by_index = {record.get("index"): record for record in other_records}
        current_only = 0
        other_only = 0
        comparable = 0
        for record in current_records:
            other = other_by_index.get(record.get("index"))
            if other is None:
                continue
            comparable += 1
            current_correct = bool(record.get("routing_correct"))
            other_correct = bool(other.get("routing_correct"))
            if current_correct and not other_correct:
                current_only += 1
            elif other_correct and not current_correct:
                other_only += 1

        discordant = current_only + other_only
        if discordant == 0:
            return {
                "comparable": comparable,
                "current_correct_other_wrong": current_only,
                "other_correct_current_wrong": other_only,
                "chi_square": None,
                "p_value": None,
            }

        chi_square = (abs(current_only - other_only) - 1) ** 2 / discordant
        p_value = math.erfc(math.sqrt(chi_square / 2))
        return {
            "comparable": comparable,
            "current_correct_other_wrong": current_only,
            "other_correct_current_wrong": other_only,
            "chi_square": chi_square,
            "p_value": p_value,
        }

    def _router_comparison(
        self,
        report_path: str,
        current_summary: Dict[str, Any],
        current_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        other_path = self._other_router_report_path(report_path)
        if not os.path.isfile(other_path):
            return {
                "available": False,
                "missing_report": other_path,
            }

        with open(other_path, "r", encoding="utf-8") as f:
            other_report = json.load(f)

        other_summary = other_report.get("summary", {})
        other_records = other_report.get("records", [])
        current_accuracy = float(current_summary.get("accuracy") or 0.0)
        other_accuracy = float(other_summary.get("accuracy") or 0.0)
        return {
            "available": True,
            "current_router": self.router_method,
            "other_router": other_report.get("router_method"),
            "current_accuracy": current_accuracy,
            "other_accuracy": other_accuracy,
            "accuracy_delta_current_minus_other": current_accuracy - other_accuracy,
            "current_total": current_summary.get("total"),
            "other_total": other_summary.get("total"),
            "mcnemar": self._mcnemar_test(current_records, other_records),
        }

    def _save_routing_report(
        self,
        report_path: str,
        answer_path: str,
        records: List[Dict[str, Any]],
    ) -> None:
        summary = self._routing_report_summary(records)
        report = {
            "experiment": self.experiment,
            "router_method": self.router_method,
            "answer_file": answer_path,
            "summary": summary,
            "router_comparison": self._router_comparison(
                report_path,
                summary,
                records,
            ),
            "records": records,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(
            "SLG routing accuracy: %.4f (%s/%s), router=%s, report=%s",
            float(summary["accuracy"]),
            int(summary["correct"]),
            int(summary["total"]),
            self.router_method,
            report_path,
        )

    def _build_graph(self):
        logger.info("Building SLG graph.")
        graph_builder = StateGraph(dict)

        graph_builder.add_node("orchestrator", self._task_analysis_node)
        graph_builder.add_node("confidence_router", self._confidence_router_node)
        graph_builder.add_node("aggregator", self._aggregator_node)

        for expert in self.expert_nodes:
            graph_builder.add_node(
                self.graph_node_by_expert[expert],
                functools.partial(
                    self._expert_node_builder,
                    model=expert,
                ),
            )

        graph_builder.add_edge(START, "orchestrator")
        graph_builder.add_conditional_edges("orchestrator", self._route_to_main_expert)

        for expert in self.expert_nodes:
            graph_builder.add_conditional_edges(
                self.graph_node_by_expert[expert],
                self._route_after_expert,
            )

        graph_builder.add_conditional_edges(
            "confidence_router", self._route_from_confidence_router
        )
        graph_builder.add_edge("aggregator", END)

        return graph_builder.compile()

    def _get_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self._build_graph()
        return self._compiled_graph

    @staticmethod
    def _initial_state(question: str) -> Dict[str, Any]:
        return {
            "question": question,
            "phase": "main",
            "selected_expert": None,
            "visited_experts": [],
            "answers": [],
            "pending_neighbors": [],
            "main_confidence": None,
            "last_expert": None,
            "routing_candidates": [],
            "routing_latency_seconds": None,
            "total_latency_seconds": None,
        }

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Run SLG for one question and return the final answer plus routing details."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        start = time.perf_counter()
        result = self._get_graph().invoke(self._initial_state(question))
        result["total_latency_seconds"] = time.perf_counter() - start
        return {
            "question": question,
            "answer": result.get("final_answer"),
            "router_method": self.router_method,
            "selected_expert": result.get("selected_expert"),
            "visited_experts": result.get("visited_experts", []),
            "candidate_answers": result.get("answers", []),
            "routing_candidates": result.get("routing_candidates", []),
            "routing_latency_seconds": result.get("routing_latency_seconds"),
            "total_latency_seconds": result.get("total_latency_seconds"),
        }

    def ask_slg(self, file: str) -> None:
        """Run the SLG graph for all questions in a file."""
        from utils.path_utils import validate_file_exists

        validate_file_exists(file)

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paths_config = CONFIG["paths"]
        output_dir = os.path.join(paths_config["answers"], self.experiment)
        ensure_dir(output_dir)
        output_basename = self._slg_output_basename()
        output_name = f"{output_basename}.json"
        output_path = os.path.join(output_dir, output_name)
        routing_report_dir = os.path.join(output_dir, "routing_reports")
        ensure_dir(routing_report_dir)
        routing_report_path = os.path.join(
            routing_report_dir,
            f"{output_basename}_routing_report.json",
        )

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

        routing_records: List[Dict[str, Any]] = []
        if os.path.exists(routing_report_path):
            with open(routing_report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            routing_records = list(report_data.get("records", []))
        elif answers_list:
            logger.warning(
                "Existing SLG answers found without routing report; rebuilding from "
                "the beginning so routing analysis is complete."
            )

        consistent_index = min(len(answers_list), len(routing_records))
        if consistent_index != len(answers_list) or consistent_index != len(routing_records):
            logger.warning(
                "SLG answer/report resume mismatch (answers=%s, routing_records=%s); "
                "resuming from %s.",
                len(answers_list),
                len(routing_records),
                consistent_index,
            )
            answers_list = answers_list[:consistent_index]
            routing_records = routing_records[:consistent_index]
            start_index = consistent_index

        graph = self._get_graph()

        for i, item in enumerate(data[start_index:], start=start_index):
            logger.info(f"Answering {i + 1}/{len(data)} questions.")
            logger.info(f"Inference of the title: {item['title']}")
            initial_state = self._initial_state(item["question"])
            question_start = time.perf_counter()
            result = graph.invoke(initial_state)
            result["total_latency_seconds"] = time.perf_counter() - question_start

            answers_list.append(
                {
                    "chapter": item["chapter"],
                    "title": item["title"],
                    "question": item["question"],
                    "answer": result.get("final_answer"),
                    "router_method": self.router_method,
                }
            )
            routing_records.append(self._build_routing_record(i, item, result))

            # Save progress incrementally so we can resume after interruptions.
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(answers_list, f, indent=4)
            self._save_routing_report(
                report_path=routing_report_path,
                answer_path=output_path,
                records=routing_records,
            )
            logger.info(40 * "-")

        return None
