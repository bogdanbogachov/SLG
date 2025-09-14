from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from config import CONFIG
from logging_config import logger
from peft import PeftModel
import json
import torch
import os
import functools
import difflib


class SmallLanguageGraph:
    def __init__(self, experts_location, experiment):
        self.experts_location = experts_location
        self.experiment = experiment

    def _categorize_task(self, prompt, experts):
        """
        Function to orchestrate questions.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Set the paths for your local model and adapter
        base_model_path = 'downloaded_3_2_1b'
        adapter_path = f"experiments/{self.experts_location}/finetuned_orchestrator_3_2_1b"

        # Load the tokenizer (from base model)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

        # Load the base model from local storage
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # Uses FP16 for lower memory usage
            device_map="auto"  # Ensures it loads to GPU automatically
        )

        # Apply the LoRA adapter on top
        # model.resize_token_embeddings(len(tokenizer)) # make sure the raw model has the same embedding size as adapter
        finetuned_model = PeftModel.from_pretrained(model, adapter_path)

        # Ensure the model is fully on GPU
        finetuned_model.to("cuda")

        # Inference
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to("cuda")
        logger.debug(f'Tokenized prompt for orchestrator: {inputs}')
        outputs = finetuned_model.generate(**inputs,
                                           max_new_tokens=10,
                                           num_return_sequences=1,
                                           temperature=0.1,
                                           eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
                                           )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = text.split("assistant")[1].strip()
        output = output.replace(' ', '_').replace('/', '_').lower()
        output = 'finetuned_' + output
        logger.info(f'Categorizer raw output: {output}')

        if output in experts:
            logger.info(f'Categorizer output found in experts list: {output}')
            return output
        else:
            closest_match = max(experts, key=lambda s: difflib.SequenceMatcher(None, output, s).ratio())
            logger.info(f'Categorizer closest match with experts: {closest_match}')
            return closest_match

    @staticmethod
    def _tuned_generate(prompt, adapter):
        logger.info("Generating from tuned")
        messages = [
            # {"role": "system", "content": CONFIG['inference_prompt']},
            {"role": "user", "content": prompt}
        ]

        # Set the paths for your local model and adapter
        base_model_path = 'downloaded_3_2_1b'
        adapter_path = adapter
        logger.info(f"Model used to infer: {adapter}")

        # Load the tokenizer (from base model)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

        # Load the base model from local storage
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # Uses FP16 for lower memory usage
            device_map="auto"  # Ensures it loads to GPU automatically
        )

        # Apply the LoRA adapter on top
        model.resize_token_embeddings(len(tokenizer))  # make sure the raw model has the same embedding size as adapter
        finetuned_model = PeftModel.from_pretrained(model, adapter_path)

        # Ensure the model is fully on GPU
        finetuned_model.to("cuda")

        # Inference
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to("cuda")
        logger.debug(f'Tokenized prompt for expert: {inputs}')
        outputs = finetuned_model.generate(**inputs,
                                           max_new_tokens=CONFIG['max_new_tokens'],
                                           num_return_sequences=1,
                                           temperature=CONFIG['temperature'],
                                           eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.debug(f"Output: {text}")
        logger.info("Inference complete.")

        del model
        del tokenizer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Helps defragment GPU memory

        return text.split("assistant")[1]

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
        experts_list_of_strings = [expert for expert in os.listdir(f'experiments/{self.experts_location}/slg')]
        response = self._categorize_task(prompt, experts_list_of_strings)
        state["category"] = response.strip().lower()
        return state

    def _expert_node_builder(self, state, model):
        """Damage classification expert node."""
        question = state["question"]
        prompt = question
        state["answer"] = self._tuned_generate(
            prompt,
            f"experiments/{self.experts_location}/slg/{model}"
        )
        return state

    def _routing_function(self, state):
        """Route based on the category identified in the task analysis."""
        models = {folder_name: folder_name for folder_name in os.listdir(f'experiments/{self.experts_location}/slg')}
        category = state.get("category")
        return models.get(category, END)

    def _build_graph(self):
        logger.info("Building graph.")
        graph_builder = StateGraph(dict)

        logger.info("Adding nodes to the graph.")
        graph_builder.add_node("task_analysis", self._task_analysis_node)
        for node in os.listdir(f'experiments/{self.experts_location}/slg'):
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
        for edge in os.listdir(f'experiments/{self.experts_location}/slg'):
            if not edge.endswith(".json"):
                logger.info(f"Adding edge {edge}.")
                graph_builder.add_edge(edge, END)

        graph = graph_builder.compile()
        logger.info("Graph built.")
        logger.info(40*'-')

        return graph

    # Step 4: Execute the Graph
    def ask_slg(self, file):
        """Run the graph for a user question."""
        # Step 1: Read the original JSON file
        with open(file, 'r') as f:
            data = json.load(f)

        # Step 2: Process the data
        graph = self._build_graph()
        answers_list = []
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

        with open(f'answers/{self.experiment}/slg.json', 'w') as f:
            json.dump(answers_list, f, indent=4)

        return None
