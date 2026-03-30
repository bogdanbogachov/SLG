"""Small Language Graph (SLG) for multi-expert question answering.

Pipeline:
1) Orchestrator/router selects the main expert.
2) Main expert generates an answer + confidence (avg token log-prob -> exp -> [0, 1]).
3) Based on confidence, we invoke 1..k neighboring experts (neighbors by cosine similarity over
   chunk embeddings computed at training time).
4) A base (non-tuned) Llama-3.2-1B-Instruct model aggregates candidate answers.
"""

import difflib
import functools
import json
import math
import os
from typing import Any, Dict, List

import torch
from langgraph.graph import END, START, StateGraph

from config import CONFIG
from logging_config import logger
from utils.model_loader import (
    cleanup_model_memory,
    load_base_model_and_tokenizer,
    load_model_with_adapter,
)
from utils.path_utils import (
    ensure_dir,
    get_slg_path,
    validate_slg_embedding_artifacts,
)
from utils.prompt_utils import (
    apply_chat_template,
    create_system_message,
    create_user_message,
)


class SmallLanguageGraph:
    def __init__(self, experts_location: str, experiment: str):
        self.experts_location = experts_location
        self.experiment = experiment

        paths_config = CONFIG["paths"]
        self.experiments_dir = paths_config["experiments"]
        self.slg_path = get_slg_path(self.experts_location, self.experiments_dir)
        validate_slg_embedding_artifacts(self.slg_path)
        self.index_path = os.path.join(self.slg_path, "index.json")

        self.slg_index = self._load_slg_index()
        self.slg_neighbors_by_expert: Dict[str, List[str]] = self.slg_index[
            "neighbors_by_expert"
        ]
        self.neighbor_k: int = int(self.slg_index["neighbor_k"])

        self.expert_nodes: List[str] = self._discover_expert_nodes()

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
                "Run commands.slg_embeddings.run_slg_embeddings for this experiment."
            )

        with open(self.index_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        neighbors_by_expert: Dict[str, List[str]] = {}
        neighbor_k = 0
        for e in entries:
            expert_id = e["expert_id"]
            top_neighbors = e.get("top_k_neighbors", [])
            neighbors_by_expert[expert_id] = list(top_neighbors)
            if neighbor_k == 0:
                neighbor_k = len(top_neighbors)

        return {"neighbors_by_expert": neighbors_by_expert, "neighbor_k": neighbor_k}

    def _categorize_task(self, prompt: str, experts: List[str]) -> str:
        """Categorize a question and route it to an appropriate expert."""
        messages = [create_user_message(prompt)]

        paths_config = CONFIG["paths"]
        models_paths = paths_config["models"]
        adapters_config = CONFIG["adapters"]

        base_model_path = os.path.join(
            paths_config["downloaded_models"], models_paths["3_2_1b"]
        )
        experiments_dir = paths_config["experiments"]
        adapter_path = os.path.join(
            experiments_dir,
            self.experts_location,
            adapters_config["orchestrator_3_2_1b"],
        )

        finetuned_model, tokenizer = load_model_with_adapter(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            resize_token_embeddings=False,
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
            orchestrator_max_tokens = generation_config["orchestrator_max_tokens"]
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=orchestrator_max_tokens,
                num_return_sequences=1,
                temperature=generation_config["temperature"],
                eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            output = (
                text.split("assistant")[1].strip()
                if "assistant" in text
                else text.strip()
            )
            output = output.replace(" ", "_").replace("/", "_").lower()

            if output in experts:
                return output

            return max(
                experts,
                key=lambda s: difflib.SequenceMatcher(None, output, s).ratio(),
            )
        finally:
            cleanup_model_memory(finetuned_model, tokenizer)

    def _task_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the task and route it to an appropriate main expert."""
        question = state["question"]

        prompt = (
            "Analyze this question and find an appropriate expert who can answer it: "
            f"{question}"
        )

        experts_list_of_strings = list(self.slg_neighbors_by_expert.keys())
        expert_set = set(self.expert_nodes)
        experts_list_of_strings = [e for e in experts_list_of_strings if e in expert_set]
        if not experts_list_of_strings:
            experts_list_of_strings = self.expert_nodes

        response = self._categorize_task(prompt, experts_list_of_strings)
        state["selected_expert"] = response.strip().lower()
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
        """Aggregate up to k+1 candidate answers into a final answer."""
        question = state["question"]
        candidates = state.get("answers", [])

        system_prompt = (
            "You are an answer aggregator.\n"
            "Combine multiple expert answers into one final answer."
        )

        candidate_lines: List[str] = []
        for i, c in enumerate(candidates, start=1):
            candidate_lines.append(
                f"{i}. Expert: {c.get('expert')}\n"
                f"   Confidence: {c.get('confidence')}\n"
                f"   Answer: {c.get('answer')}"
            )

        user_prompt = (
            f"Question:\n{question}\n\n"
            "Candidate answers:\n"
            + "\n".join(candidate_lines)
            + "\n\n"
            "Instructions:\n"
            "- Combine overlapping correct information into one answer.\n"
            "- Prefer the most consistent and well-supported claims.\n"
            "- If answers conflict, choose the best-supported one or mention uncertainty.\n"
            "- Do not mention the internal process unless needed.\n"
            "- Return only the final answer."
        )

        messages = [
            create_system_message(system_prompt),
            create_user_message(user_prompt),
        ]

        paths_config = CONFIG["paths"]
        models_paths = paths_config["models"]
        base_model_path = os.path.join(
            paths_config["downloaded_models"], models_paths["3_2_1b"]
        )

        model, tokenizer = load_base_model_and_tokenizer(base_model_path)
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
            aggregation_max_new_tokens = generation_config.get(
                "aggregation_max_new_tokens", 256
            )

            eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            outputs = model.generate(
                **inputs,
                max_new_tokens=aggregation_max_new_tokens,
                num_return_sequences=1,
                temperature=generation_config["temperature"],
                eos_token_id=eos_id,
                do_sample=False,
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            state["final_answer"] = self._extract_assistant_text(decoded)
            return state
        finally:
            cleanup_model_memory(model, tokenizer)

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

        graph = self._build_graph()
        answers_list: List[Dict[str, Any]] = []

        for item in data:
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
            logger.info(40 * "-")

        output_path = os.path.join(output_dir, "slg.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(answers_list, f, indent=4)

        return None
