"""Application-wide constants."""

# Directory paths
DIR_EXPERIMENTS = 'experiments'
DIR_ANSWERS = 'answers'
DIR_QUESTION_ANSWER = 'question_answer'
DIR_CHECKPOINTS = 'checkpoints'
DIR_DOWNLOADED_MODELS = 'downloaded_models'
DIR_SPLIT_BY_TITLE = 'question_answer/split_by_title'
DIR_CHARTS = 'experiments/charts'

# Model directory names (subdirectories of downloaded_models)
MODEL_DIR_3_2_1B = 'downloaded_3_2_1b'
MODEL_DIR_3_1_8B = 'downloaded_3_1_8b'

# File paths
FILE_QA_ORIGINAL = 'question_answer/qa_original.json'
FILE_QA_INFLATING_MATERIAL = 'question_answer/inflating_material.json'
FILE_QA = 'question_answer/qa.json'
FILE_QA_TRAIN = 'question_answer/qa_train.json'
FILE_QA_TEST = 'question_answer/qa_test.json'
FILE_SRM_PDF = 'question_answer/srm.pdf'
FILE_OM_PDF = 'question_answer/om.pdf'

# Training constants
DEFAULT_TEST_SPLIT_RATIO = 0.20
DEFAULT_MAX_LENGTH = 1024
DEFAULT_NUM_QUESTIONS = 28
DEFAULT_PREFIX_LENGTH = 5

# RAG/Embedding constants
EMBEDDING_DIMENSION = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RAG_BATCH_SIZE = 100
RAG_K_RETRIEVALS = 5

# Model generation constants
DEFAULT_MAX_NEW_TOKENS = 750
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.1
DEFAULT_ORCHESTRATOR_MAX_TOKENS = 10

# Logging
DEFAULT_LOG_DIR = 'logs'
DEFAULT_LOG_FILE = 'eng_llm.log'

