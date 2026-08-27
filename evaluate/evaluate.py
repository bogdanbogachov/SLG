import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import ast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
import re
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import time
from typing import Optional
from openai import OpenAI
import numpy as np

from logging_config import logger
from config import CONFIG


logger.propagate = False

_EVAL_CLIENT = None
_WORDNET_READY = None

def load_data(predictions_file, ground_truth_file):
    """
    Load predictions and ground truth answers from JSON files.
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    return predictions, ground_truth


def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score between a reference and a candidate answer.
    """
    reference_tokens = reference.split()  # BLEU expects a list of references
    candidate_tokens = candidate.split()
    smoothing = SmoothingFunction().method4  # For smoothing BLEU scores
    return sentence_bleu(references=reference_tokens, hypothesis=candidate_tokens, smoothing_function=smoothing)


def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE scores between a reference and a candidate answer.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target=reference, prediction=candidate)
    return scores


def calculate_exact_match(reference, candidate):
    """
    Calculate Exact Match (EM) score between a reference and a candidate answer.
    """
    return int(reference.strip() == candidate.strip())


# Function to calculate METEOR scores
def calculate_meteor_score(reference, candidate):
    """
    Calculate meteor score between a reference and a candidate answer.
    """
    try:
        return meteor_score(references=[word_tokenize(reference)], hypothesis=word_tokenize(candidate))
    except LookupError:
        if _ensure_wordnet():
            return meteor_score(references=[word_tokenize(reference)], hypothesis=word_tokenize(candidate))
        raise


def _ensure_wordnet():
    global _WORDNET_READY
    if _WORDNET_READY is not None:
        return _WORDNET_READY

    for resource in ('corpora/wordnet', 'corpora/wordnet.zip'):
        try:
            nltk.data.find(resource)
            _WORDNET_READY = True
            return _WORDNET_READY
        except LookupError:
            continue

    _WORDNET_READY = nltk.download('wordnet', quiet=True)
    return _WORDNET_READY


def get_embedding(text: str, client) -> np.ndarray:
    """
    Get embedding for text using OpenAI API.
    
    Args:
        text: Text to embed
        client: OpenAI client instance
        
    Returns:
        Embedding vector as numpy array
    """
    models_config = CONFIG['models']
    embedding_model = models_config['embedding_model']
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )

    return np.array(response.data[0].embedding)


def check_entailment(reference: str, candidate: str, api_client) -> float:
    """
    Calculates cosine similarity between the OpenAI embeddings for reference and candidate.
    Returns:
        Cosine similarity score (float, range -1 to 1).
    """
    # Get embeddings for each text
    emb_ref = get_embedding(reference, api_client)
    emb_cand = get_embedding(candidate, api_client)

    # Compute cosine similarity
    similarity = np.dot(emb_ref, emb_cand) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_cand))
    # Normalize from 0 to 1
    similarity = (similarity + 1) / 2

    return similarity


def _aggregate_metrics(bleu_scores, rouge_scores, exact_matches, meteor_scores, entailment_scores, ai_experts):
    """Turn per-question score lists into averaged metrics dict."""
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {
        key: sum(rouge_scores[key]) / len(rouge_scores[key])
        for key in rouge_scores.keys()
    } if any(rouge_scores.values()) else {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    avg_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_entailment = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0
    avg_ai_expert = sum(ai_experts) / len(ai_experts) if ai_experts else 0
    return {
        'BLEU': avg_bleu,
        'ROUGE': avg_rouge,
        'Exact Match': avg_exact_match,
        'METEOR': avg_meteor,
        'Entailment': avg_entailment,
        'AI Expert': avg_ai_expert,
    }


def _empty_score_lists():
    return (
        [],
        {'rouge1': [], 'rouge2': [], 'rougeL': []},
        [],
        [],
        [],
        [],
    )


def _score_records_to_lists(score_records):
    bleu_scores, rouge_scores, exact_matches, meteor_scores, entailment_scores, ai_experts = _empty_score_lists()
    for scores in score_records:
        bleu_scores.append(scores['BLEU'])
        for key in rouge_scores.keys():
            rouge_scores[key].append(scores['ROUGE'][key])
        exact_matches.append(scores['Exact Match'])
        meteor_scores.append(scores['METEOR'])
        entailment_scores.append(scores['Entailment'])
        ai_experts.append(scores['AI Expert'])
    return bleu_scores, rouge_scores, exact_matches, meteor_scores, entailment_scores, ai_experts


def _aggregate_score_records(score_by_index):
    ordered_scores = [
        score_by_index[i]
        for i in sorted(score_by_index.keys())
    ]
    return _aggregate_metrics(*_score_records_to_lists(ordered_scores))


def _checkpoint_from_v1_state(state: dict):
    next_i = int(state['next_i'])
    bleu_scores = state['bleu_scores']
    rouge_scores = state['rouge_scores']
    exact_matches = state['exact_matches']
    meteor_scores = state['meteor_scores']
    entailment_scores = state['entailment_scores']
    ai_experts = state['ai_experts']

    score_count = len(bleu_scores)
    lengths = {
        score_count,
        len(rouge_scores.get('rouge1', [])),
        len(rouge_scores.get('rouge2', [])),
        len(rouge_scores.get('rougeL', [])),
        len(exact_matches),
        len(meteor_scores),
        len(entailment_scores),
        len(ai_experts),
    }
    if lengths != {next_i}:
        raise ValueError(
            'legacy checkpoint cannot be converted because score lists do not match next_i'
        )

    completed_scores = {}
    for i in range(next_i):
        completed_scores[i] = {
            'BLEU': bleu_scores[i],
            'ROUGE': {
                'rouge1': rouge_scores['rouge1'][i],
                'rouge2': rouge_scores['rouge2'][i],
                'rougeL': rouge_scores['rougeL'][i],
            },
            'Exact Match': exact_matches[i],
            'METEOR': meteor_scores[i],
            'Entailment': float(entailment_scores[i]),
            'AI Expert': ai_experts[i],
        }
    return completed_scores, set()


def _save_eval_checkpoint(
    path: str,
    n_pairs: int,
    score_by_index,
    skipped_indices,
):
    """Persist progress so evaluation can resume after interruption."""
    partial_metrics = _aggregate_score_records(score_by_index)
    state = {
        'version': 2,
        'n_pairs': n_pairs,
        'completed_scores': {
            str(i): score_by_index[i]
            for i in sorted(score_by_index.keys())
        },
        'skipped_indices': sorted(skipped_indices),
        'partial_metrics': partial_metrics,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def _load_eval_checkpoint(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _load_eval_progress(path: str, n_pairs: int):
    state = _load_eval_checkpoint(path)
    if state.get('n_pairs') != n_pairs:
        logger.warning(
            'Checkpoint does not match current data length (%s vs %s); starting evaluation from scratch.',
            state.get('n_pairs'),
            n_pairs,
        )
        return {}, set()

    version = state.get('version', 1)
    if version == 1:
        return _checkpoint_from_v1_state(state)
    if version != 2:
        raise ValueError(f'unsupported evaluation checkpoint version: {version}')

    completed_scores = {
        int(i): scores
        for i, scores in state.get('completed_scores', {}).items()
    }
    skipped_indices = {
        int(i)
        for i in state.get('skipped_indices', [])
    }
    return completed_scores, skipped_indices


def _normalise_answer(value):
    if value == '' or value is None:
        return 'Empty'
    return value


def _init_eval_worker(api_key: str):
    global _EVAL_CLIENT
    _EVAL_CLIENT = OpenAI(api_key=api_key)


def _get_eval_client():
    global _EVAL_CLIENT
    if _EVAL_CLIENT is None:
        _EVAL_CLIENT = OpenAI(api_key=CONFIG['open_ai_api_key'])
    return _EVAL_CLIENT


def _score_eval_pair(task):
    i, pred, truth = task
    if pred['question'] != truth['question']:
        return {
            'index': i,
            'matched': False,
            'chapter': pred.get('chapter'),
            'title': pred.get('title'),
            'pred_question': pred['question'],
            'truth_question': truth['question'],
        }

    gt_answer = _normalise_answer(truth['answer'])
    pred_answer = _normalise_answer(pred['answer'])
    client = _get_eval_client()

    bleu = calculate_bleu(gt_answer, pred_answer)
    rouge = calculate_rouge(gt_answer, pred_answer)
    exact_match = calculate_exact_match(gt_answer, pred_answer)
    meteor = calculate_meteor_score(gt_answer, pred_answer)
    entailment = check_entailment(gt_answer, pred_answer, api_client=client)
    ai_expert = calculate_ai_expert(gt_answer, pred_answer, api_client=client)

    return {
        'index': i,
        'matched': True,
        'scores': {
            'BLEU': bleu,
            'ROUGE': {
                'rouge1': rouge['rouge1'].fmeasure,
                'rouge2': rouge['rouge2'].fmeasure,
                'rougeL': rouge['rougeL'].fmeasure,
            },
            'Exact Match': exact_match,
            'METEOR': meteor,
            'Entailment': float(entailment),
            'AI Expert': ai_expert,
        },
    }


def calculate_ai_expert(reference, candidate, api_client):
    """
    Calculate AI expert scores between a reference and a candidate answer using OpenAI GPT-4.1 Nano.
    """
    ai_expert_prompt = CONFIG['ai_expert_prompt']
    query_expert_prompt = CONFIG["query_expert_prompt"]
    max_new_tokens = CONFIG['generation']['max_new_tokens']
    temperature = CONFIG['generation']['temperature']

    try:
        models_config = CONFIG['models']
        response = api_client.chat.completions.create(
            model=models_config['gpt_4_1_nano'],
            messages=[
                {"role": "system", "content": ai_expert_prompt},
                {"role": "user", "content": query_expert_prompt.format(text_1=reference, text_2=candidate)},
            ],
            max_completion_tokens=max_new_tokens,
            temperature=temperature
        )
        llm_response = response.choices[0].message.content.strip()
        time.sleep(1)
    except Exception as e:
        logger.info(f"API call failed: {e}")
        return 0

    try:
        return int(llm_response)
    except Exception as e:
        logger.info(f"Could not convert response to int: '{llm_response}' -- {e}")
        return 0


def _record_eval_result(result, score_by_index, skipped_indices):
    i = result['index']
    if result['matched']:
        score_by_index[i] = result['scores']
        return

    skipped_indices.add(i)
    logger.info(
        "Warning: questions didn't match at chapter: %s, and title: %s",
        result['chapter'],
        result['title'],
    )
    logger.info('Pred question: %s', result['pred_question'])
    logger.info('Truth question: %s', result['truth_question'])
    logger.info(40 * '-')


def _configured_eval_workers(eval_workers: Optional[int], n_pairs: int) -> int:
    eval_cfg = CONFIG.get('evaluation') or {}
    if eval_workers is None:
        eval_workers = eval_cfg.get('workers', 1)
    try:
        workers = int(eval_workers)
    except (TypeError, ValueError):
        logger.warning('Invalid evaluation worker count %r; using 1 worker.', eval_workers)
        workers = 1
    if workers < 1:
        workers = 1
    if n_pairs > 0:
        workers = min(workers, n_pairs)
    return workers


def _log_eval_progress(processed_count: int, n_pairs: int, threshold: int) -> int:
    percent_checked = (processed_count / n_pairs) * 100 if n_pairs else 100
    if threshold < percent_checked:
        logger.info('Evaluated %s%%', threshold)
        threshold += 10
    return threshold


def evaluate(
    predictions,
    ground_truth,
    checkpoint_path: Optional[str] = None,
    eval_workers: Optional[int] = None,
):
    """
    Evaluate predictions against ground truth using BLEU, ROUGE, and Exact Match.

    If ``checkpoint_path`` is set, progress is saved periodically and the run resumes
    from the last checkpoint after an interruption. The checkpoint is removed when
    evaluation finishes successfully. Delete the checkpoint file manually to start over.
    """
    eval_cfg = CONFIG.get('evaluation') or {}
    checkpoint_every = int(eval_cfg.get('checkpoint_every', 10))
    if checkpoint_every < 1:
        checkpoint_every = 1

    n_pairs = min(len(predictions), len(ground_truth))
    workers = _configured_eval_workers(eval_workers, n_pairs)
    score_by_index = {}
    skipped_indices = set()

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            score_by_index, skipped_indices = _load_eval_progress(checkpoint_path, n_pairs)
            logger.info(
                'Resuming evaluation with %s/%s questions already processed (checkpoint).',
                len(score_by_index) + len(skipped_indices),
                n_pairs,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning('Could not load evaluation checkpoint (%s); starting from scratch.', e)

    pending_tasks = [
        (i, predictions[i], ground_truth[i])
        for i in range(n_pairs)
        if i not in score_by_index and i not in skipped_indices
    ]

    logger.info('Evaluation has started with %s worker process(es).', workers)
    threshold = 10
    processed_count = len(score_by_index) + len(skipped_indices)
    threshold = _log_eval_progress(processed_count, n_pairs, threshold)
    processed_since_checkpoint = 0

    def maybe_save_checkpoint(force: bool = False):
        if not checkpoint_path:
            return
        if not force and processed_since_checkpoint < checkpoint_every:
            return
        _save_eval_checkpoint(
            checkpoint_path,
            n_pairs,
            score_by_index,
            skipped_indices,
        )
        logger.info(
            'Saved evaluation checkpoint (%s/%s questions processed).',
            len(score_by_index) + len(skipped_indices),
            n_pairs,
        )

    if pending_tasks:
        if workers == 1:
            _init_eval_worker(CONFIG['open_ai_api_key'])
            for task in pending_tasks:
                result = _score_eval_pair(task)
                _record_eval_result(result, score_by_index, skipped_indices)
                processed_count = len(score_by_index) + len(skipped_indices)
                threshold = _log_eval_progress(processed_count, n_pairs, threshold)
                processed_since_checkpoint += 1
                if processed_since_checkpoint >= checkpoint_every:
                    maybe_save_checkpoint()
                    processed_since_checkpoint = 0
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_eval_worker,
                initargs=(CONFIG['open_ai_api_key'],),
            ) as executor:
                futures = [
                    executor.submit(_score_eval_pair, task)
                    for task in pending_tasks
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception:
                        maybe_save_checkpoint(force=True)
                        raise
                    _record_eval_result(result, score_by_index, skipped_indices)
                    processed_count = len(score_by_index) + len(skipped_indices)
                    threshold = _log_eval_progress(processed_count, n_pairs, threshold)
                    processed_since_checkpoint += 1
                    if processed_since_checkpoint >= checkpoint_every:
                        maybe_save_checkpoint()
                        processed_since_checkpoint = 0

    if pending_tasks and processed_since_checkpoint > 0:
        maybe_save_checkpoint(force=True)

    result = _aggregate_score_records(score_by_index)
    logger.info('Evaluation has been completed.')
    logger.info(40 * '-')

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info('Removed evaluation checkpoint (completed).')
        except OSError as e:
            logger.warning('Could not remove evaluation checkpoint: %s', e)

    return result


def extract_log_values(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
        content = re.sub(r'\bnan\b', "100000", content)
        content = re.sub(r'\binf\b', "100000", content)
        data = _extract_last_log_history(content)

        second_last_log = data[-2]  # Get the second-to-last dictionary
        last_log = data[-1]  # Get the last dictionary

        return {
            'train_loss': second_last_log['train_loss'],
            'train_runtime': second_last_log['train_runtime'],
            'eval_loss': last_log['eval_loss'],
            'epochs': last_log['epoch']
        }


def _extract_list_literals(content: str):
    literals = []
    depth = 0
    start = None
    quote = None
    escape = False

    for index, char in enumerate(content):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
            continue

        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                literals.append(content[start:index + 1])
                start = None

    return literals


def _extract_last_log_history(content: str):
    for literal in reversed(_extract_list_literals(content)):
        try:
            data = ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            continue
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            return data

    raise ValueError("Could not parse trainer log history list.")


def pull_training_metrics(base_folder):
    metrics = []
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if folder == 'slg':
            all_train_loss = 0
            all_eval_loss = 0
            total_train_runtime = 0
            total_epochs = 0
            count = 0

            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    training_log_path = os.path.join(subfolder_path, 'training_log.txt')
                    if os.path.exists(training_log_path):
                        logger.debug(training_log_path)
                        log_values = extract_log_values(training_log_path)
                        all_train_loss += log_values['train_loss']
                        all_eval_loss += log_values['eval_loss']
                        total_train_runtime += log_values['train_runtime']
                        total_epochs += log_values['epochs']
                        count += 1


            metrics.append(
                {
                    'avg_train_loss_slg': all_train_loss / count,
                    'avg_eval_loss_slg': all_eval_loss / count,
                    'total_train_runtime_slg': total_train_runtime,
                    'avg_epochs_slg': total_epochs / count
                }
            )
        elif folder != 'slg' and 'logs' not in folder:
            training_log_path = os.path.join(folder_path, 'training_log.txt')
            logger.debug(training_log_path)
            if os.path.exists(training_log_path):
                log_values = extract_log_values(training_log_path)
                metrics.append(
                    {
                        f'train_loss_{folder}': log_values['train_loss'],
                        f'all_eval_loss_{folder}': log_values['eval_loss'],
                        f'train_runtime_{folder}': log_values['train_runtime'],
                        f'epochs_{folder}': log_values['epochs']
                    }
                )

    return metrics
