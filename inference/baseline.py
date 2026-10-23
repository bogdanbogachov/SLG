"""Inference functions for baseline, RAG, and fine-tuned models."""
from transformers import AutoTokenizer
import numpy as np
import json
import os
from typing import Dict, Any, List

from logging_config import logger
from config import CONFIG
from utils.model_loader import load_model_with_adapter, cleanup_model_memory
from utils.path_utils import ensure_dir


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
    import time
    import torch
    
    validate_file_exists(file)
    
    with open(file, 'r') as f:
        data = json.load(f)

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
    output_file = os.path.join(output_dir, f"{os.path.basename(adapter)}.jsonl")

    # Load existing progress if available
    answers = {}
    if os.path.exists(output_file):
        with open(output_file, encoding='utf-8') as f:
            for line in f:
                try:
                    if line.strip():
                        record = json.loads(line)
                        if record.get('qid'):
                            answers[record['qid']] = record
                except json.JSONDecodeError:
                    logger.warning("Ignoring an incomplete final inference line while resuming.")
        logger.info("Resuming fine-tuned inference: %s/%s qids already present.", len(answers), len(data))
        with open(output_file, 'w', encoding='utf-8') as repaired:
            for row in data:
                record = answers.get(row.get('qid'))
                if record is not None:
                    repaired.write(json.dumps(record, ensure_ascii=False) + '\n')
    pending = [(i, item) for i, item in enumerate(data) if item.get('qid') not in answers]
    
    try:
        with open(output_file, 'a', encoding='utf-8') as out_file:
          for i, item in pending:
            logger.info(f'Answering {i + 1}/{len(data)} questions.')
            started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            messages = [create_user_message(item['question'])]
            
            # Apply chat template
            prompt = apply_chat_template(messages, tokenizer, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to("cuda")
            logger.debug(f'Tokenized prompt for baseline fine-tuned: {inputs}')
            
            generation_config = CONFIG['generation']
            # PeftModel.generate() forwards to the inner model; `generator=` is rejected by
            # Transformers' model_kwarg validation on some versions. Global RNG is still
            # seeded via config import (set_project_seed).
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=generation_config['max_new_tokens'],
                num_return_sequences=1,
                do_sample=False,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output = text.split("assistant")[1] if "assistant" in text else text

            new_dict = {
                "qid": item.get("qid"),
                "chapter": item['chapter'],
                "title": item['title'],
                "question": item['question'],
                "answer": output,
                "latency_s": time.perf_counter() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
            }
            answers[item.get('qid')] = new_dict
            out_file.write(json.dumps(new_dict, ensure_ascii=False) + '\n')
            out_file.flush()
    
    finally:
        # Always cleanup memory
        cleanup_model_memory(finetuned_model, tokenizer)
    
    return None


class AskRag:
    """Local FAISS retrieval with the base Llama 3.1 8B Instruct generator."""
    def __init__(self, documents_file: str, questions_file: str, experiment: str, model_path: str):
        from utils.path_utils import validate_file_exists
        validate_file_exists(documents_file)
        validate_file_exists(questions_file)
        self.documents_file = documents_file
        self.questions_file = questions_file
        self.experiment = experiment
        self.model_path = model_path

    def _retrieve_documents(self):
        import faiss
        from sentence_transformers import SentenceTransformer
        cfg = CONFIG['rag']
        with open(self.documents_file, encoding='utf-8') as f:
            rows = json.load(f)
        corpus = list(dict.fromkeys(r['answer'] for r in rows if r.get('answer')))
        size, overlap = int(cfg['chunk_size']), int(cfg['chunk_overlap'])
        chunks = []
        for answer in corpus:
            words = answer.split()
            chunks.extend(' '.join(words[i:i + size]) for i in range(0, max(1, len(words)), size - overlap) if words[i:i + size])
        paths = CONFIG['paths']
        encoder_path = os.path.join(paths['downloaded_models'], paths['models']['jina_embeddings'])
        encoder = SentenceTransformer(encoder_path, trust_remote_code=True)
        vectors = encoder.encode(chunks, batch_size=int(cfg['batch_size']), convert_to_numpy=True, show_progress_bar=False).astype('float32')
        index = faiss.IndexFlatL2(int(cfg['embedding_dimension']))
        index.add(vectors)
        return chunks, index, encoder

    def generate_responses(self):
        import gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        with open(self.questions_file, encoding='utf-8') as f:
            rows = json.load(f)
        out_dir = os.path.join(CONFIG['paths']['answers'], self.experiment)
        ensure_dir(out_dir)
        output_path = os.path.join(out_dir, 'rag.jsonl')
        completed = {}
        if os.path.exists(output_path):
            with open(output_path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if item.get('qid'):
                            completed[item['qid']] = item
            with open(output_path, 'w', encoding='utf-8') as repaired:
                for row in rows:
                    record = completed.get(row.get('qid'))
                    if record is not None:
                        repaired.write(json.dumps(record, ensure_ascii=False) + '\n')
        pending = [r for r in rows if r.get('qid') not in completed]
        if not pending:
            return list(completed.values())
        chunks, index, encoder = self._retrieve_documents()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        tokenizer.padding_side = "left"
        device_map = {'': 0} if torch.cuda.is_available() else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=device_map, local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        device = model.get_input_embeddings().weight.device
        cfg = CONFIG['rag']
        gen = CONFIG['generation']
        batch_size = int(cfg.get('generation_batch_size', 4))
        with open(output_path, 'a', encoding='utf-8') as out:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                logger.info(f'Answering RAG {offset + 1}-{offset + len(batch)}/{len(pending)} questions.')
                query_vectors = encoder.encode([r['question'] for r in batch], convert_to_numpy=True).astype('float32')
                distances, indices = index.search(query_vectors, int(cfg['k_retrievals']))
                prompts = []
                start_times = []
                for row, hits in zip(batch, indices):
                    context = ' '.join(chunks[int(i)] for i in hits if i >= 0)
                    messages = [
                        {'role': 'system', 'content': CONFIG['rag_prompt']},
                        {'role': 'user', 'content': f"Context: {context}\nQuestion: {row['question']}\nAnswer:"},
                    ]
                    prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                    start_times.append(__import__('time').perf_counter())
                inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=int(gen['max_new_tokens']), do_sample=False, pad_token_id=tokenizer.pad_token_id)
                texts = tokenizer.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
                for row, answer, started in zip(batch, texts, start_times):
                    record = {k: row.get(k) for k in ('qid', 'chapter', 'title', 'question')}
                    record.update(answer=answer.strip(), latency_s=__import__('time').perf_counter() - started, peak_gpu_memory_mb=peak_mb)
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
        return list(completed.values())
