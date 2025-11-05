"""Inference functions for baseline, RAG, and fine-tuned models."""
from transformers import AutoTokenizer
import numpy as np
import faiss
import torch
import json
import os
from typing import Dict, Any, List

from logging_config import logger
from config import CONFIG
from utils.model_loader import load_model_with_adapter, cleanup_model_memory
from utils.path_utils import ensure_dir


def ask_baseline(file: str, model: str, experiment: str, client) -> None:
    """
    Generate responses using baseline model (OpenAI API).
    Supports resume on failure.
    
    Args:
        file: Path to JSON file with questions
        model: Model name (from CONFIG)
        experiment: Experiment name for output directory
        client: OpenAI client instance
    """
    from utils.path_utils import validate_file_exists, ensure_dir
    
    validate_file_exists(file)
    
    logger.info(f"Asking baseline model: {model}.")

    with open(file, 'r') as f:
        data = json.load(f)

    paths_config = CONFIG['paths']
    output_dir = os.path.join(paths_config['answers'], experiment)
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"{model}.json")

    # Load existing progress if available
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            answers_list = json.load(f)
        start_index = len(answers_list)
        logger.info(f"Resuming from index {start_index}/{len(data)}.")
    else:
        answers_list = []
        start_index = 0
        logger.info("Starting fresh run.")

    for index, item in enumerate(data[start_index:], start=start_index):
        logger.info(f'Answering {index} out of {len(data)} questions.')

        try:
            generation_config = CONFIG['generation']
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CONFIG['inference_prompt']},
                    {"role": "user", "content": item['question']},
                ],
                max_tokens=generation_config['max_new_tokens'],
                temperature=generation_config['temperature']
            )
            llm_response = response.choices[0].message.content.strip()

        except Exception as e:
            logger.info(f"API call failed at index {index}: {e}")
            llm_response = "API call failed."

        new_dict = {
            "chapter": item['chapter'],
            "title": item['title'],
            "question": item['question'],
            "answer": llm_response
        }
        answers_list.append(new_dict)

        # Save progress incrementally
        with open(output_path, 'w') as f:
            json.dump(answers_list, f, indent=4)

    logger.info("All questions processed.")
    return None


def ask_finetuned(file: str, base_model: str, adapter: str, experiment: str) -> None:
    """
    Generate responses using a fine-tuned model.
    
    Args:
        file: Path to JSON file with questions
        base_model: Path to base model directory
        adapter: Path to adapter directory
        experiment: Experiment name for output directory
    """
    from utils.path_utils import validate_file_exists
    from utils.prompt_utils import apply_chat_template, create_user_message
    
    validate_file_exists(file)
    
    with open(file, 'r') as f:
        data = json.load(f)

    answers: List[Dict[str, Any]] = []
    logger.info(f"Model used to infer: {adapter}")

    # Load model with adapter using utility function
    finetuned_model, tokenizer = load_model_with_adapter(
        base_model_path=base_model,
        adapter_path=adapter,
        resize_token_embeddings=False
    )
    
    paths_config = CONFIG['paths']
    output_dir = os.path.join(paths_config['answers'], experiment)
    ensure_dir(output_dir)
    
    try:
        for item in data:
            messages = [create_user_message(item['question'])]
            
            # Apply chat template
            prompt = apply_chat_template(messages, tokenizer, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to("cuda")
            logger.debug(f'Tokenized prompt for baseline fine-tuned: {inputs}')
            
            generation_config = CONFIG['generation']
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=generation_config['max_new_tokens'],
                num_return_sequences=1,
                temperature=generation_config['temperature'],
                eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
            )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output = text.split("assistant")[1] if "assistant" in text else text

            new_dict = {
                "chapter": item['chapter'],
                "title": item['title'],
                "question": item['question'],
                "answer": output
            }
            answers.append(new_dict)

            # Save incrementally
            output_file = os.path.join(output_dir, f"{adapter.split('/')[-1]}.json")
            with open(output_file, 'w') as f:
                json.dump(answers, f, indent=4)
    
    finally:
        # Always cleanup memory
        cleanup_model_memory(finetuned_model, tokenizer)
    
    return None


class AskRag:
    """
    This is used to operate RAG.
    """
    def __init__(self, documents_file: str, questions_file: str, experiment: str, client):
        """
        Initialize RAG system.
        
        Args:
            documents_file: Path to documents JSON file
            questions_file: Path to questions JSON file
            experiment: Experiment name
            client: OpenAI client instance
        """
        from utils.path_utils import validate_file_exists
        
        validate_file_exists(documents_file)
        validate_file_exists(questions_file)
        
        self.documents_file = documents_file
        self.questions_file = questions_file
        self.experiment = experiment
        self.client = client

    def _retrieve_documents(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Retrieves documents from passed in question-answer pairs file.
        Splits long documents into chunks and embeds each chunk.
        """
        rag_config = CONFIG['rag']
        if chunk_size is None:
            chunk_size = rag_config['chunk_size']
        if chunk_overlap is None:
            chunk_overlap = rag_config['chunk_overlap']

        logger.info("Asking RAG.")
        with open(self.documents_file, 'r') as file:
            data = json.load(file)

        documents = [document['answer'] for document in data if document['answer'] != ""]
        models_config = CONFIG['models']
        embedding_model = models_config['embedding_model']

        embedding_dimension = rag_config['embedding_dimension']
        index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance index

        def chunk_text(text: str, size: int = chunk_size, overlap: int = chunk_overlap):
            words = text.split()
            for j in range(0, len(words), size - overlap):
                yield " ".join(words[j:j + size])

        chunked_docs = []
        doc_ids = []  # keep mapping to original document index
        for doc_id, doc in enumerate(documents):
            for chunk in chunk_text(doc):
                chunked_docs.append(chunk)
                doc_ids.append(doc_id)

        logger.info(f"Prepared {len(chunked_docs)} chunks from {len(documents)} documents.")

        rag_batch_size = rag_config['batch_size']
        for i in range(0, len(chunked_docs), rag_batch_size):
            batch = chunked_docs[i:i + rag_batch_size]
            response = self.client.embeddings.create(
                model=embedding_model,
                input=batch
            )
            batch_embeddings = np.array(
                [item.embedding for item in response.data],
                dtype="float32"
            )
            index.add(batch_embeddings)

        logger.info("RAG has finished embedding chunks.")

        # Return both original docs and chunks (with mapping)
        return chunked_docs, index

    @staticmethod
    def _retrieve_answers(query: str, documents: List[str], index, client, k: int = None) -> List[str]:
        """
        Retrieval model.
        """
        if k is None:
            rag_config = CONFIG['rag']
            k = rag_config['k_retrievals']

        models_config = CONFIG['models']
        embedding_model = models_config['embedding_model']
        response = client.embeddings.create(model=embedding_model, input=query)

        query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
        distances, indices = index.search(query_embedding, k)

        return [documents[i] for i in indices[0]]

    def generate_responses(self):
        """
        Pass question and retrieved docs to an LLM to aggregate a response.
        Supports resume on failure (continues where it left off).
        """
        with open(self.questions_file, 'r') as file:
            data = json.load(file)

        from utils.path_utils import ensure_dir
        
        paths_config = CONFIG['paths']
        output_dir = os.path.join(paths_config['answers'], self.experiment)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, "rag.json")

        # Load existing progress if available
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                answers_list = json.load(f)
            start_index = len(answers_list)
            logger.info(f"Resuming from index {start_index}/{len(data)}.")
        else:
            answers_list = []
            start_index = 0
            logger.info("Starting fresh run.")

        # Build FAISS index once (not on every resume)
        documents, index = self._retrieve_documents()

        for i, item in enumerate(data[start_index:], start=start_index):
            logger.info(f"Answering {i + 1}/{len(data)}")

            rag_config = CONFIG['rag']
            retrieved_docs = self._retrieve_answers(
                query=item['question'],
                documents=documents,
                index=index,
                client=self.client,
                k=rag_config['k_retrievals']
            )

            context = " ".join(retrieved_docs)
            input_text = f"Context: {context}\nQuestion: {item['question']}\nAnswer:"

            try:
                models_config = CONFIG['models']
                generation_config = CONFIG['generation']
                response = self.client.chat.completions.create(
                    model=models_config['gpt_4_1_nano'],
                    messages=[
                        {"role": "system", "content": CONFIG['rag_prompt']},
                        {"role": "user", "content": input_text},
                    ],
                    max_tokens=generation_config['max_new_tokens'],
                    temperature=generation_config['temperature']
                )
                llm_response = response.choices[0].message.content.strip()
            except Exception as e:
                logger.info(f"API call failed at index {i}: {e}")
                llm_response = "API call failed."

            new_dict = {
                "chapter": item['chapter'],
                "title": item['title'],
                "question": item['question'],
                "answer": llm_response
            }
            answers_list.append(new_dict)

            # Save progress incrementally
            with open(output_path, 'w') as f:
                json.dump(answers_list, f, indent=4)

        logger.info("All questions processed.")
        return answers_list
