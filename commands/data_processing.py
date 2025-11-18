import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from logging_config import logger
from config import CONFIG


def create_qa():
    from question_answer.question_answer_gen import populate
    from question_answer.srm_reader import read_doc
    from question_answer.om_reader import prepare_overhaul_manual

    files_config = CONFIG['files']
    srm_pdf = files_config['srm_pdf']
    om_pdf = files_config['om_pdf']
    
    df_srm = read_doc(srm_pdf)
    populate(df_srm, 'srm_qa')

    df_om = prepare_overhaul_manual(overhaul_manual=om_pdf)
    populate(df_om, 'om_qa')


def combine_all_qa():
    paths_config = CONFIG['paths']
    files_config = CONFIG['files']
    folder_path = paths_config['question_answer']

    all_qa = []

    for filename in os.listdir(folder_path):
        if 'qa' in filename and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    all_qa.extend(data)
                except Exception as e:
                    logger.info(f"Error reading {filename}: {e}")

    with open(files_config['qa'], 'w', encoding='utf-8') as out_file:
        json.dump(all_qa, out_file, indent=4)

    return None


def inflate_overshadowing():
    from question_answer.inflate_overshadowing import inflate_qa_answers_with_file_inputs
    files_config = CONFIG['files']
    inflate_qa_answers_with_file_inputs(
        files_config['qa_original'],
        files_config['qa_inflating_material'],
        files_config['qa']
    )


def split_train_test(data_file):
    # Load JSON file
    logger.info(f"Splitting {data_file} into train and test sets.")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Delete all instances where title == answer
    data = [entry for entry in data if entry.get("title") != entry.get("answer")]

    # Split data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save train and test data
    files_config = CONFIG['files']
    with open(files_config['qa_train'], "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)

    with open(files_config['qa_test'], "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    logger.info("Train and test splitting is complete.")

    return None


def split_qa_pairs_by_title(data_file):
    # Load the JSON data
    logger.info(f"Splitting {data_file} by title.")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group entries by unique "title"
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["title"]].append(entry)

    # Save each group as a separate JSON file
    paths_config = CONFIG['paths']
    split_by_title_path = paths_config['split_by_title']
    os.makedirs(split_by_title_path, exist_ok=True)
    
    for title, entries in grouped_data.items():
        # Create a valid filename by replacing spaces and special characters
        title = title.replace(' ', '_').replace('/', '_').replace('\n', '_').lower()
        filename = f"{title}.json"

        with open(os.path.join(split_by_title_path, filename), "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent=4, ensure_ascii=False)

    logger.info(f"Title splitting is complete.")

    return None


def split_qa():
    files_config = CONFIG['files']
    split_train_test(files_config['qa'])
    split_qa_pairs_by_title(files_config['qa_train'])


def data_overlap_check():
    from evaluate.data_overlap import compute_overshadowing
    data_config = CONFIG['data']
    prefix_length = data_config['prefix_length']
    compute_overshadowing(prefix_length=prefix_length)
