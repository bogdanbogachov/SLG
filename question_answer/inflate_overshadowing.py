import json
import hashlib
import math
from typing import List, Dict, Set
from collections import Counter


def sort_json_by_title_and_answer(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as infile:
        data: List[Dict] = json.load(infile)

    # Count how often each title appears
    title_counts = Counter(item.get("title", "") for item in data)

    # Sort by: (title frequency, title alphabetically, answer alphabetically)
    sorted_data = sorted(
        data,
        key=lambda item: (
            title_counts[item.get("title", "")],
            item.get("title", ""),
            item.get("answer", "")
        )
    )

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(sorted_data, outfile, indent=4)

    return None


def _validate_inflation_percentage(inflation_percentage: float) -> float:
    percentage = float(inflation_percentage)
    if percentage < 0 or percentage > 100:
        raise ValueError("inflation_percentage must be between 0 and 100.")
    return percentage


def _stable_answer_rank(title: str, answer: str, seed: int) -> int:
    payload = f"{seed}\0{title}\0{answer}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _select_answers_to_inflate(
    title: str,
    answers: List[str],
    inflation_percentage: float,
    seed: int,
) -> Set[str]:
    if inflation_percentage == 0 or not answers:
        return set()

    if inflation_percentage == 100:
        return set(answers)

    answers_to_inflate = math.floor((len(answers) * inflation_percentage / 100) + 0.5)
    ranked_answers = sorted(
        answers,
        key=lambda answer: (_stable_answer_rank(title, answer, seed), answer),
    )
    return set(ranked_answers[:answers_to_inflate])


def inflate_qa_answers_with_file_inputs(
    qa_original_path: str,
    inflating_path: str,
    qa_output_path: str,
    inflation_percentage: float = 100,
    seed: int = 42,
):
    inflation_percentage = _validate_inflation_percentage(inflation_percentage)

    # Sort the QAs by title and answer
    sort_json_by_title_and_answer(qa_original_path, qa_output_path)

    # Load data from files
    with open(qa_output_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    with open(inflating_path, "r", encoding="utf-8") as f:
        inflating_data = json.load(f)

    inflating_texts = [item["text"] for item in inflating_data]
    inflating_count = len(inflating_texts)

    # Group QAs by title
    title_to_qas = {}
    for entry in qa_data:
        title = entry.get("title", "")
        title_to_qas.setdefault(title, []).append(entry)

    # Assign inflation per distinct answer per title
    for title in sorted(title_to_qas.keys()):
        answer_to_inflation = {}

        for qa in title_to_qas[title]:
            answer_text = qa["answer"]

            if answer_text not in answer_to_inflation:
                if len(answer_to_inflation) >= inflating_count:
                    raise IndexError(f"Not enough inflating texts for title: {title}")
                answer_to_inflation[answer_text] = inflating_texts[len(answer_to_inflation)]

        answers_to_inflate = _select_answers_to_inflate(
            title=title,
            answers=list(answer_to_inflation.keys()),
            inflation_percentage=inflation_percentage,
            seed=seed,
        )

        for qa in title_to_qas[title]:
            answer_text = qa["answer"]
            if answer_text not in answers_to_inflate:
                continue

            inflating_text = answer_to_inflation[answer_text]
            qa["answer"] = f"{inflating_text}\n\n{answer_text}"

    # Flatten back into a list
    result = [qa for qas in title_to_qas.values() for qa in qas]

    with open(qa_output_path, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=4)

    return None
