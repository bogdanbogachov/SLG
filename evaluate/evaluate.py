import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
import re
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import time
from openai import OpenAI
import numpy as np

from logging_config import logger
from config import CONFIG


# Download WordNet for synonym matching
nltk.download('wordnet')

logger.propagate = False

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
    score = meteor_score(references=[word_tokenize(reference)], hypothesis=word_tokenize(candidate))
    return score


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


def calculate_ai_expert(reference, candidate, api_client):
    """
    Calculate AI expert scores between a reference and a candidate answer using OpenAI GPT-4.1 Nano.
    """
    ai_expert_prompt = CONFIG['ai_expert_prompt']
    query_expert_prompt = CONFIG["query_expert_prompt"]
    max_new_tokens = CONFIG['max_new_tokens']
    temperature = CONFIG['temperature']

    try:
        models_config = CONFIG['models']
        response = api_client.chat.completions.create(
            model=models_config['gpt_4_1_nano'],
            messages=[
                {"role": "system", "content": ai_expert_prompt},
                {"role": "user", "content": query_expert_prompt.format(text_1=reference, text_2=candidate)},
            ],
            max_tokens=max_new_tokens,
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


def evaluate(predictions, ground_truth):
    """
    Evaluate predictions against ground truth using BLEU, ROUGE, and Exact Match.
    """

    # OpenAI api key
    client = OpenAI(api_key=CONFIG['open_ai_api_key'])

    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    exact_matches = []
    meteor_scores = []
    entailment_scores = []
    ai_experts = []
    logger.info(f"Evaluation has started.")
    counter = 0
    threshold = 10
    for pred, truth in zip(predictions, ground_truth):
        # if counter in [i for i in range(0, len(ground_truth), 30)]:
        #     time.sleep(45)
        counter += 1
        percent_checked = (counter/len(predictions))*100
        if threshold < percent_checked:
            logger.info(f"Evaluated {threshold}%")
            threshold += 10
        if pred['question'] != truth['question']:
            logger.info(f"Warning: questions didn't match at chapter: {pred['chapter']}, and title: {pred['title']}")
            logger.info(f"Pred question: {pred['question']}")
            logger.info(f"Truth question: {truth['question']}")
            logger.info(40*'-')
            continue

        if truth['answer'] == '':
            gt_answer = 'Empty'
        elif truth['answer'] is None:
            gt_answer = 'Empty'
        else:
            gt_answer = truth['answer']

        if pred['answer'] == '':
            pred_answer = 'Empty'
        elif pred['answer'] is None:
            pred_answer = 'Empty'
        else:
            pred_answer = pred['answer']

        # BLEU
        logger.debug("Calculating BLEU score.")
        bleu = calculate_bleu(gt_answer, pred_answer)
        bleu_scores.append(bleu)

        # ROUGE
        logger.debug("Calculating ROUGE score.")
        rouge = calculate_rouge(gt_answer, pred_answer)
        for key in rouge_scores.keys():
            rouge_scores[key].append(rouge[key].fmeasure)

        # Exact Match
        logger.debug("Calculating Exact Match (EM) score.")
        exact_matches.append(calculate_exact_match(gt_answer, pred_answer))

        # Meteor
        logger.debug("Calculating METEOR score.")
        meteor = calculate_meteor_score(gt_answer, pred_answer)
        meteor_scores.append(meteor)

        # Entailment
        logger.debug("Calculating entailment score.")
        entailment = check_entailment(gt_answer, pred_answer, api_client=client)
        entailment_scores.append(entailment)

        # AI expert
        logger.debug("Calculating AI expert score.")
        ai_expert = calculate_ai_expert(gt_answer, pred_answer, api_client=client)
        ai_experts.append(ai_expert)

    # Aggregate scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {key: sum(rouge_scores[key]) / len(rouge_scores[key]) for key in rouge_scores.keys()}
    avg_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_entailment = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0
    avg_ai_expert = sum(ai_experts) / len(ai_experts) if ai_experts else 0
    logger.info(f"Evaluation has been completed.")
    logger.info(40*'-')

    return {
        "BLEU": avg_bleu,
        "ROUGE": avg_rouge,
        "Exact Match": avg_exact_match,
        "METEOR": avg_meteor,
        "Entailment": avg_entailment,
        "AI Expert": avg_ai_expert
    }


def extract_log_values(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
        content = re.sub(r'\bnan\b', "100000", content)
        content = re.sub(r'\binf\b', "100000", content)
        data = json.loads(content.replace("'", '"'))

        second_last_log = data[-2]  # Get the second-to-last dictionary
        last_log = data[-1]  # Get the last dictionary

        return {
            'train_loss': second_last_log['train_loss'],
            'train_runtime': second_last_log['train_runtime'],
            'eval_loss': last_log['eval_loss'],
            'epochs': last_log['epoch']
        }


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
