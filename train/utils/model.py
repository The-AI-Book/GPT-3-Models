from openai.cli import FineTune
import pandas as pd
import json
import utils.settings as st

class ArgsDatasetObject():
    def __init__(self, file, quiet):
        self.file = file
        self.quiet = quiet

class ArgsCreateObject():
    def __init__(self, 
            training_file, 
            validation_file, 
            model, 
            n_epochs, 
            batch_size,
            learning_rate_multiplier, 
            prompt_loss_weight
        ):
        self.training_file = training_file
        self.validation_file = validation_file
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate_multiplier = learning_rate_multiplier
        self.prompt_loss_weight = prompt_loss_weight
        self.check_if_files_exist = True 
        self.suffix = None
        self.compute_classification_metrics = None
        self.classification_n_classes = None
        self.classification_positive_class = None
        self.classification_betas = None
        self.no_follow = None

        
def approximate_costs(data: pd.DataFrame, validation_size: float):
    tokens = 0
    for text in data["completion"].values:
        tokens += len(text.split(" "))
    num_tokens = tokens * (1-validation_size) 
    training_costs = {
        f"Number of tokens": num_tokens,
        f"Estimated ADA cost (USD)": round(0.0008 * num_tokens / 2, 3),
        f"Estimated Babbage cost (USD)": round(0.0012 * num_tokens / 2),
        f"Estimated Curie cost (USD)": round(0.0060 * num_tokens / 2),
        f"Estimated Davinci cost (USD)": round(0.0600 * num_tokens / 2)
    }
    print(training_costs)
    with open(st.COSTS_PATH, 'w') as fd:
        json.dump(
            training_costs,
            fd, indent=4
        )

def prepare_data(data_path: str):
    """
    Prepare data using FineTune prepare data method of OpenAI Library.
    """
    args = ArgsDatasetObject(
        file = data_path, 
        quiet = True
    )
    FineTune.prepare_data(args)

def train_model():
    args = ArgsCreateObject(
        training_file=st.TRAIN_DATA_PATH_JSONL, 
        validation_file=st.VALIDATION_DATA_PATH_JSONL, 
        model = st.MODEL,
        n_epochs= st.N_EPOCHS,
        batch_size= st.BATCH_SIZE, 
        learning_rate_multiplier= st.LEARNING_RATE_MULTIPLIER,
        prompt_loss_weight= st.PROMPT_LOSS_WEIGHT
    )
    FineTune.create(args)

if __name__ == "__main__":
    print("Hello world")