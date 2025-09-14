from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
import faiss
import torch
import json
import os

from logging_config import logger
from config import CONFIG


def ask_baseline(file, model, experiment, client):
    """
    Generic flow to ask LLM questions with resume support.
    If the process fails, it can pick up from the last completed index.
    """

    logger.info(f"Asking baseline model: {model}.")

    with open(file, 'r') as f:
        data = json.load(f)

    output_path = f"answers/{experiment}/{model}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CONFIG['inference_prompt']},
                    {"role": "user", "content": item['question']},
                ],
                max_tokens=CONFIG['max_new_tokens'],
                temperature=CONFIG['temperature']
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


def ask_finetuned(file, base_model, adapter, experiment):
    """
    Generates a response by finetuned baseline model.
    """
    with open(file, 'r') as f:
        data = json.load(f)

    answers = []
    logger.info(f"Model used to infer: {adapter}")

    # Set the paths for your local model and adapter
    base_model_path = base_model
    adapter_path = adapter

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

    for item in data:
        messages = [
            {"role": "user", "content": item['question']}
        ]

        # Create the pipeline
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to("cuda")
        logger.debug(f'Tokenized prompt for baseline fine-tuned: {inputs}')
        outputs = finetuned_model.generate(**inputs,
                                           max_new_tokens=CONFIG['max_new_tokens'],
                                           num_return_sequences=1,
                                           temperature=CONFIG['temperature'],
                                           eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = text.split("assistant")[1]

        new_dict = {
            "chapter": item['chapter'],
            "title": item['title'],
            "question": item['question'],
            "answer": output
        }
        answers.append(new_dict)

        with open(f"answers/{experiment}/{adapter.split('/')[-1]}.json", 'w') as f:
            json.dump(answers, f, indent=4)

    # Purge all GPU memory after inference
    del model
    del tokenizer
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Helps defragment GPU memory

    return None


class AskRag:
    """
    This is used to operate RAG.
    """
    def __init__(self, documents_file, questions_file, experiment, client):
        self.documents_file = documents_file
        self.questions_file = questions_file
        self.experiment = experiment
        self.client = client

    def _retrieve_documents(self, chunk_size=500, chunk_overlap=50):
        """
        Retrieves documents from passed in question-answer pairs file.
        Splits long documents into chunks and embeds each chunk.
        """

        logger.info("Asking RAG.")
        with open(self.documents_file, 'r') as file:
            data = json.load(file)

        documents = [document['answer'] for document in data if document['answer'] != ""]
        embedding_model = CONFIG['embedding_model']

        dimension = 1536
        index = faiss.IndexFlatL2(dimension)  # L2 distance index

        def chunk_text(text, size=chunk_size, overlap=chunk_overlap):
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

        batch_size = 100

        for i in range(0, len(chunked_docs), batch_size):
            batch = chunked_docs[i:i + batch_size]
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
    def _retrieve_answers(query, documents, index, client, k=5):
        """
        Retrieval model.
        """

        response = client.embeddings.create(model=CONFIG['embedding_model'], input=query)

        query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
        distances, indices = index.search(np.array(query_embedding), k)

        return [documents[i] for i in indices[0]]

    def generate_responses(self):
        """
        Pass question and retrieved docs to an LLM to aggregate a response.
        Supports resume on failure (continues where it left off).
        """
        with open(self.questions_file, 'r') as file:
            data = json.load(file)

        output_path = f"answers/{self.experiment}/rag.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
        documents, faiss_index = self._retrieve_documents()

        for i, item in enumerate(data[start_index:], start=start_index):
            logger.info(f"Answering {i + 1}/{len(data)}")

            retrieved_docs = self._retrieve_answers(
                query=item['question'],
                documents=documents,
                faiss_index=faiss_index,
                client=self.client,
                k=5
            )

            context = " ".join(retrieved_docs)
            input_text = f"Context: {context}\nQuestion: {item['question']}\nAnswer:"

            try:
                response = self.client.chat.completions.create(
                    model=CONFIG["gpt_4_1_nano"],
                    messages=[
                        {"role": "system", "content": CONFIG['rag_prompt']},
                        {"role": "user", "content": input_text},
                    ],
                    max_tokens=CONFIG['max_new_tokens'],
                    temperature=CONFIG['temperature']
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
