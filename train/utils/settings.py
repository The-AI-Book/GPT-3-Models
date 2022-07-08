import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

COSTS_PATH = "./data/costs.json"
RANDOM_SAMPLE_SIZE = 2000
RAW_DATA_PATH = "./data/raw.txt"
RAW_DATA_PROCESSED_PATH = "./data/raw_processed.csv"
TRAIN_DATA_PATH = "./data/train_data.csv"
VALIDATION_DATA_PATH = "./data/validation_data.csv"
TRAIN_DATA_PATH_JSONL = "./data/train_data_prepared.jsonl"
VALIDATION_DATA_PATH_JSONL = "./data/validation_data_prepared.jsonl"
MODEL = 'curie'  # can be ada, babbage or curie
N_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE_MULTIPLIER = 0.1
PROMPT_LOSS_WEIGHT = 0.1

PROMPT_NAME = "Juan Esteban 🐸"
COMPLETION_NAME = "Madre"

CHAT_STOP_WORDS = ["<Multimedia omitido>", "(archivo adjunto)", 
                    "[Video]", "Llamada perdida"]
PROJECT_NAME = "bring-me-back"

PROMPT_WORD = " ->"
STOP_WORD = " END"