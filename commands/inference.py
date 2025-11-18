import os
from openai import OpenAI
from logging_config import logger
from config import CONFIG
from utils.path_utils import ensure_dir


def run_baseline(experiment: str):
    from inference.baseline import ask_baseline
    client = OpenAI(api_key=CONFIG['open_ai_api_key'])
    paths_config = CONFIG['paths']
    answers_dir = paths_config['answers']
    ensure_dir(os.path.join(answers_dir, experiment))
    files_config = CONFIG['files']
    model = CONFIG['models']['gpt_4_1']
    ask_baseline(file=files_config['qa_test'], model=model, experiment=experiment, client=client)


def run_rag(experiment: str):
    from inference.baseline import AskRag
    client = OpenAI(api_key=CONFIG['open_ai_api_key'])
    files_config = CONFIG['files']
    rag = AskRag(
        documents_file=files_config['qa_train'],
        questions_file=files_config['qa_test'],
        experiment=experiment,
        client=client
    )
    rag.generate_responses()


def run_finetuned(experiment: str):
    from inference.baseline import ask_finetuned
    paths_config = CONFIG['paths']
    models_paths = paths_config['models']
    adapters_config = CONFIG['adapters']
    answers_dir = paths_config['answers']
    experiments_dir = paths_config['experiments']
    downloaded_models_dir = paths_config['downloaded_models']
    
    ensure_dir(os.path.join(answers_dir, experiment))
    files_config = CONFIG['files']
    
    base_model_3_2_1b = os.path.join(downloaded_models_dir, models_paths['3_2_1b'])
    base_model_3_1_8b = os.path.join(downloaded_models_dir, models_paths['3_1_8b'])
    
    ask_finetuned(file=files_config['qa_test'],
                  base_model=base_model_3_2_1b,
                  adapter=os.path.join(experiments_dir, experiment, adapters_config['finetuned_3_2_1b']),
                  experiment=experiment)
    ask_finetuned(file=files_config['qa_test'],
                  base_model=base_model_3_1_8b,
                  adapter=os.path.join(experiments_dir, experiment, adapters_config['finetuned_3_1_8b']),
                  experiment=experiment)


def run_slg(experiment: str):
    from inference.slg import SmallLanguageGraph
    paths_config = CONFIG['paths']
    answers_dir = paths_config['answers']
    ensure_dir(os.path.join(answers_dir, experiment))
    files_config = CONFIG['files']
    slg = SmallLanguageGraph(experts_location=experiment, experiment=experiment)
    slg.ask_slg(file=files_config['qa_test'])
