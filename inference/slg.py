"""Small Language Graph (SLG) for multi-expert question answering."""
from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer
from config import CONFIG
from logging_config import logger
import json
import os
import functools
import difflib
from typing import Dict, Any, List

from utils.model_loader import load_model_with_adapter, cleanup_model_memory
from utils.prompt_utils import apply_chat_template, create_user_message
from utils.path_utils import ensure_dir, get_slg_path


class SmallLanguageGraph:
    def __init__(self, experts_location, experiment):
        self.experts_location = experts_location
        self.experiment = experiment

    def _categorize_task(self, prompt: str, experts: List[str]) -> str:
        """
        Categorize a question and route it to an appropriate expert.
        
        Args:
            prompt: The question prompt
            experts: List of available expert names
            
        Returns:
            Name of the expert to use
        """
        import torch
        
        messages = [create_user_message(prompt)]

        # Set the paths for your local model and adapter
        paths_config = CONFIG['paths']
        models_paths = paths_config['models']
        adapters_config = CONFIG['adapters']
        base_model_path = os.path.join(
            paths_config['downloaded_models'],
            models_paths['3_2_1b']
        )
        experiments_dir = paths_config['experiments']
        adapter_path = os.path.join(experiments_dir, self.experts_location, adapters_config['orchestrator_3_2_1b'])

        # Load model with adapter using utility function
        finetuned_model, tokenizer = load_model_with_adapter(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            resize_token_embeddings=False
        )

        try:
            # Inference
            formatted_prompt = apply_chat_template(messages, tokenizer, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=False, truncation=True).to("cuda")
            logger.debug(f'Tokenized prompt for orchestrator: {inputs}')
            
            generation_config = CONFIG['generation']
            orchestrator_max_tokens = generation_config['orchestrator_max_tokens']
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=orchestrator_max_tokens,
                num_return_sequences=1,
                temperature=generation_config['temperature'],
                eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output = text.split("assistant")[1].strip() if "assistant" in text else text.strip()
            output = output.replace(' ', '_').replace('/', '_').lower()
            logger.info(f'Categorizer raw output: {output}')

            if output in experts:
                logger.info(f'Categorizer output found in experts list: {output}')
                return output
            else:
                closest_match = max(experts, key=lambda s: difflib.SequenceMatcher(None, output, s).ratio())
                logger.info(f'Categorizer closest match with experts: {closest_match}')
                return closest_match
        finally:
            # Always cleanup memory
            cleanup_model_memory(finetuned_model, tokenizer)

    @staticmethod
    def _tuned_generate(prompt: str, adapter: str) -> str:
        """
        Generate a response using a fine-tuned expert model.
        
        Args:
            prompt: The question prompt
            adapter: Path to the adapter directory
            
        Returns:
            Generated response text
        """
        logger.info("Generating from tuned")
        messages = [create_user_message(prompt)]

        # Set the paths for your local model and adapter
        paths_config = CONFIG['paths']
        models_paths = paths_config['models']
        base_model_path = os.path.join(
            paths_config['downloaded_models'],
            models_paths['3_2_1b']
        )
        adapter_path = adapter
        logger.info(f"Model used to infer: {adapter}")

        # Load model with adapter using utility function
        finetuned_model, tokenizer = load_model_with_adapter(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            resize_token_embeddings=True  # SLG experts need this
        )

        try:
            # Inference
            formatted_prompt = apply_chat_template(messages, tokenizer, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=False, truncation=True).to("cuda")
            logger.debug(f'Tokenized prompt for expert: {inputs}')
            
            generation_config = CONFIG['generation']
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=generation_config['max_new_tokens'],
                num_return_sequences=1,
                temperature=generation_config['temperature'],
                eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            logger.debug(f"Output: {text}")
            logger.info("Inference complete.")

            return text.split("assistant")[1] if "assistant" in text else text
        finally:
            # Always cleanup memory
            cleanup_model_memory(finetuned_model, tokenizer)

    # Step 2: Define Node Functions
    def _task_analysis_node(self, state):
        """Analyze the task and route to appropriate expert."""
        question = state["question"]
        # experts_list = "\n".join(
        #     f"- {expert}" for expert in os.listdir(f'experiments/{self.experts_location}/slg'))
        prompt = (f"Analyze this question and find an appropriate expert who can answer it: {question}"
            # f"Analyze this question and find an appropriate expert who can answer it:\n {question}"
            # f"A friendly reminder, experts are as follows:\n {experts_list}"
            # f"Very important! Return only an expert name, nothing else!"
        )
        paths_config = CONFIG['paths']
        experiments_dir = paths_config['experiments']
        slg_path = get_slg_path(self.experts_location, experiments_dir)
        experts_list_of_strings = [expert for expert in os.listdir(slg_path) if not expert.endswith('.json')]
        response = self._categorize_task(prompt, experts_list_of_strings)
        state["category"] = response.strip().lower()
        return state

    def _expert_node_builder(self, state, model):
        """Damage classification expert node."""
        question = state["question"]
        prompt = question
        paths_config = CONFIG['paths']
        experiments_dir = paths_config['experiments']
        slg_path = get_slg_path(self.experts_location, experiments_dir)
        state["answer"] = self._tuned_generate(
            prompt,
            os.path.join(slg_path, model)
        )
        return state

    def _routing_function(self, state: Dict[str, Any]):
        """
        Route based on the category identified in the task analysis.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name or END
        """
        paths_config = CONFIG['paths']
        experiments_dir = paths_config['experiments']
        slg_path = get_slg_path(self.experts_location, experiments_dir)
        models = {
            folder_name: folder_name 
            for folder_name in os.listdir(slg_path) 
            if not folder_name.endswith('.json')
        }
        category = state.get("category")
        return models.get(category, END)

    def _build_graph(self):
        logger.info("Building graph.")
        graph_builder = StateGraph(dict)

        paths_config = CONFIG['paths']
        experiments_dir = paths_config['experiments']
        slg_path = get_slg_path(self.experts_location, experiments_dir)
        
        logger.info("Adding nodes to the graph.")
        graph_builder.add_node("task_analysis", self._task_analysis_node)
        for node in os.listdir(slg_path):
            if not node.endswith(".json"):
                logger.info(f"Adding node {node}.")
                graph_builder.add_node(
                    node, functools.partial(
                        self._expert_node_builder,
                        model=node
                    )
                )

        logger.info("Adding edges to the graph.")
        graph_builder.add_edge(START, "task_analysis")
        graph_builder.add_conditional_edges("task_analysis", self._routing_function)
        for edge in os.listdir(slg_path):
            if not edge.endswith(".json"):
                logger.info(f"Adding edge {edge}.")
                graph_builder.add_edge(edge, END)

        graph = graph_builder.compile()
        logger.info("Graph built.")
        logger.info(40*'-')

        return graph

    def ask_slg(self, file: str) -> None:
        """
        Run the SLG graph for all questions in a file.
        
        Args:
            file: Path to JSON file with questions
        """
        from utils.path_utils import validate_file_exists
        
        validate_file_exists(file)
        
        # Read the original JSON file
        with open(file, 'r') as f:
            data = json.load(f)

        # Ensure output directory exists
        paths_config = CONFIG['paths']
        output_dir = os.path.join(paths_config['answers'], self.experiment)
        ensure_dir(output_dir)

        # Process the data
        graph = self._build_graph()
        answers_list: List[Dict[str, Any]] = []
        
        for item in data:
            logger.info(f"Inference of the title: {item['title']}")
            initial_state = {"question": item['question']}
            result = graph.invoke(initial_state)

            new_dict = {
                "chapter": item['chapter'],
                "title": item['title'],
                "question": item['question'],
                "answer": result.get("answer")
            }
            answers_list.append(new_dict)
            logger.info(40*'-')

        output_path = os.path.join(output_dir, 'slg.json')
        with open(output_path, 'w') as f:
            json.dump(answers_list, f, indent=4)

        return None
