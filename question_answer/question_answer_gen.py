from openai import OpenAI
import json
import time
import os

from logging_config import logger
from config import CONFIG


def generate(text):
    """
        Generates questions for ground truth texts.
        Args:
            - text: The text to generate questions for.
        Returns:
            - a tuple with all generated questions.
    """
    client = OpenAI(api_key=CONFIG['open_ai_api_key'])
    
    models_config = CONFIG['models']
    system_prompt = CONFIG['system_prompt']
    query_prompt = CONFIG["query_prompt"]
    generation_config = CONFIG['generation']
    max_new_tokens = generation_config['max_new_tokens']
    temperature = generation_config['temperature']

    questions = tuple()
    # An average word has 4 letters, an average sentence has 20 words. Thus, I divide texts by 80 to get the number
    # of sentences. The hypothesis behind this is that each sentence should have a question.
    # number_of_questions = max(1, math.ceil(int(len(text)/(4*20))))
    number_of_questions = 28 # 28 was empirically proven to be enough questions
    logger.info(f"Number of questions: {number_of_questions}")
    for i in range(0, number_of_questions):
        logger.info(f"Working on question # {i}")
        response = client.chat.completions.create(
            model=models_config['gpt_4_1_nano'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_prompt.format(text=text, questions=questions)}
            ],
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        llm_response = response.choices[0].message.content.strip()
        questions += (llm_response,)

    return questions


def populate(df, qa_file_name):
    index_to_start_qa_gen = 0

    # Check if the file exists
    if not os.path.exists(f"question_answer/{qa_file_name}.json"):
        # If the file does not exist, initialize it as an empty list
        with open(f"question_answer/{qa_file_name}.json", "w") as json_file:
            json.dump([], json_file, indent=4)
    else:
        # Load JSON data
        json_file_path = f"question_answer/{qa_file_name}.json"
        with open(json_file_path, "r") as file:
            data = json.load(file)  # Assumes the JSON file contains a list of dictionaries

        # Get the last dictionary
        last_dict = data[-1]

        # Specify the key whose value you want to compare
        key_to_compare = "answer"

        last_value = last_dict[key_to_compare]

        # Find the index of the last identical value in reverse order
        last_identical_index = len(data) - 1
        for i in range(len(data) - 2, -1, -1):  # Start from second-to-last element
            if data[i][key_to_compare] == last_value:
                last_identical_index = i
            else:
                break

        # Delete all these values in the JSON file
        data = data[:last_identical_index]

        # Save the updated JSON file
        with open(json_file_path, "w") as file:
            json.dump(data, file, indent=4)

        # Find the new last value in the last dictionary
        if len(data) > 0:
            last_dict = data[-1]  # Updated last dictionary
            new_last_value = last_dict[key_to_compare]

            # Find the last occurrence of the new_last_value in a specific column
            column_to_search = "text"
            index_to_start_qa_gen = df[df[column_to_search] == new_last_value].index[-1] + 1

    for index, row in df.iterrows():
        if index >= index_to_start_qa_gen:
            logger.info(f"Working on text # {index}")
            questions = generate(row['text'])
            for i, question in enumerate(questions):
                new_data = {
                    'chapter': row['chapter'],
                    'title': row['title'],
                    'question': question,
                    'answer': row['text']
                }

                # Now, open the file in read-write mode
                with open(f"question_answer/{qa_file_name}.json", "r+") as json_file:
                    # Load the existing data
                    try:
                        existing_data = json.load(json_file)
                    except json.JSONDecodeError:
                        existing_data = []  # Handle empty or invalid file

                    # Append the new dictionary
                    existing_data.append(new_data)

                    # Move to the beginning of the file and write updated data
                    json_file.seek(0)
                    json.dump(existing_data, json_file, indent=4)
                    json_file.truncate()  # Remove any leftover content

            logger.info(40*'-')

    return None
