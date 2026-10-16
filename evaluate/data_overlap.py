import os
import json
import numpy as np
from typing import List
from collections import defaultdict

from logging_config import logger


# Load answers from a JSON file
def unpack_json_answers(file: str) -> List[str]:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [entry['answer'] for entry in data if 'answer' in entry]

def count_overshadowing_prefixes(file, prefix_length):
    prefix_map = defaultdict(set)
    answers = unpack_json_answers(file)
    answers = list(set(answers))
    for text in answers:
        words = text.split()
        prefix = ' '.join(words[:prefix_length])
        continuation = ' '.join(words[prefix_length:]) if len(words) > prefix_length else ''
        prefix_map[prefix].add(continuation)

    # Count overshadowing
    unique_overshadowing_prefixes = sum(1 for continuations in prefix_map.values() if len(continuations) > 1)
    total_overshadowing_cases = sum(len(continuations)
                                        for continuations in prefix_map.values() if len(continuations) > 1)

    return unique_overshadowing_prefixes, total_overshadowing_cases

def compute_overshadowing(prefix_length, split_by_title_dir=None, train_file=None):
    # Check texts which title splits
    split_title_overshadowing_prefixes_unique = []
    split_title_overshadowing_cases_total = []
    if split_by_title_dir is None or train_file is None:
        from config import CONFIG
        split_by_title_dir = split_by_title_dir or CONFIG["paths"]["split_by_title"]
        train_file = train_file or CONFIG["files"]["qa_train"]
    folder_path = split_by_title_dir
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            unique, total = count_overshadowing_prefixes(file_path, prefix_length=prefix_length)
            split_title_overshadowing_prefixes_unique.append(unique)
            split_title_overshadowing_cases_total.append(total)

    avg_split_title_overshadowing_prefixes_unique = np.mean(split_title_overshadowing_prefixes_unique)
    avg_split_title_overshadowing_cases_total = np.mean(split_title_overshadowing_cases_total)
    logger.info(f"Avg unique overshadowing prefixes in split texts: {avg_split_title_overshadowing_prefixes_unique}")
    logger.info(f"Avg total overshadowing cases in split texts: {avg_split_title_overshadowing_cases_total}")

    # Check train text
    train_file_path = train_file
    train_dataset_overshadowing_prefixes_unique, train_dataset_overshadowing_cases_total\
        = count_overshadowing_prefixes(train_file_path,prefix_length=prefix_length)
    logger.info(f"Train dataset unique overshadowing prefixes: "
                f"{train_dataset_overshadowing_prefixes_unique}")
    logger.info(f"Train dataset total overshadowing cases: "
                f"{train_dataset_overshadowing_cases_total}")

    return None
