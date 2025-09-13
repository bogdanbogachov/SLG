from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login
import numpy as np
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
        logger.debug(f'Answering {index} out of {len(data)} questions.')

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

    with open(f"answers/{experiment}/{model}.json", 'w') as f:
        json.dump(answers_list, f, indent=4)
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
                                           max_new_tokens=750,
                                           num_return_sequences=1,
                                           temperature=0.1,
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
    def __init__(self, documents_file, questions_file, experiment):
        self.documents_file = documents_file
        self.questions_file = questions_file
        self.experiment = experiment

    def _retrieve_documents(self):
        """
        Retrieves documents from passed in question-answer pairs file.
        """
        import faiss

        logger.info("Asking RAG.")
        with open(self.documents_file, 'r') as file:
            data = json.load(file)

        documents = [document['answer'] for document in data if document['answer'] != ""]

        retrieval_model = SentenceTransformer(CONFIG['retrieval_model'])
        dimension = 384  # Embedding size of the sentence transformer
        index = faiss.IndexFlatL2(dimension)  # L2 distance index

        logger.info("RAG has started to embed documents.")
        doc_embeddings = retrieval_model.encode(documents)
        index.add(np.array(doc_embeddings))
        logger.info("RAG has finished to embed documents.")

        return documents, index

    @staticmethod
    def _retrieve_answers(query, documents, index, k=15):
        """
        Retrieval model.
        """
        retrieval_model = SentenceTransformer(CONFIG['retrieval_model'])

        query_embedding = retrieval_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = index.search(np.array(query_embedding), k)

        return [documents[i] for i in indices[0]]

    def generate_responses(self):
        """
        Pass question and retreived docs to an LLM to aggregate a response.
        """
        login(CONFIG['api_key'])
        client = InferenceClient()
        model = CONFIG['3_3_70b']
        tokenizer = AutoTokenizer.from_pretrained(model)

        with open(self.questions_file, 'r') as file:
            data = json.load(file)

        answers_list = []
        documents, index = self._retrieve_documents()
        for item in data:
            retrieved_docs = self._retrieve_answers(query=item['question'], documents=documents, index=index)
            context = " ".join(retrieved_docs)
            input_text = f"Context: {context}\nQuestion: {item['question']}\nAnswer:"

            messages = [
                {"role": "system", "content": CONFIG['rag_prompt']},
                {"role": "user", "content": input_text}
            ]

            total = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            llm_response = client.text_generation(
                total,
                model=model,
                max_new_tokens=CONFIG['max_new_tokens'],
                seed=CONFIG['seed'],
                temperature=CONFIG['temperature']
            )

            new_dict = {
                "chapter": item['chapter'],
                "title": item['title'],
                "question": item['question'],
                "answer": llm_response
            }
            answers_list.append(new_dict)

        with open(f"answers/{self.experiment}/rag.json", 'w') as f:
            json.dump(answers_list, f, indent=4)

        # Decode and return the response
        return answers_list
