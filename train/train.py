import wandb
import utils.preprocess as pp
import utils.settings as st
import os
import pandas as pd
import openai
from tqdm import tqdm
from utils.model import train_model

def main():
    # Get model parameters.
    config = {
        f"MODEL": st.MODEL,
        f"N_EPOCHS": st.N_EPOCHS,
        f"BATCH_SIZE": st.BATCH_SIZE,
        f"LEARNING_RATE_MULTIPLIER": st.LEARNING_RATE_MULTIPLIER,
        f"PROMPT_LOSS_WEIGHT": st.PROMPT_LOSS_WEIGHT
    }

    # Init wandb.
    wandb.init(project = st.PROJECT_NAME, 
               job_type = "logging_dataset_as_table",
               config = config)

    # Run preprocess functions. 
    data = pp.txt_to_csv_2(
        st.RAW_DATA_PATH, 
        st.RAW_DATA_PROCESSED_PATH
    )

    pp.get_train_and_validation_data(
        st.RAW_DATA_PROCESSED_PATH, 
        st.TRAIN_DATA_PATH, 
        st.VALIDATION_DATA_PATH, 
        validation_size=0.1, 
        save_files = True
    )

    wandb.run.log({"dataset": data})
    wandb.finish()

    # Train the model.
    train_model()

def validate():
    # create eval job
    os.system("openai wandb sync --project {name}".format(name = st.PROJECT_NAME))
    run = wandb.init(project=st.PROJECT_NAME, job_type='eval')
    entity = wandb.run.entity
    artifact_job = run.use_artifact('{entity}/{project}/fine_tune_details:latest'.format(entity = entity, project = st.PROJECT_NAME), type='fine_tune_details')
    wandb.config.update({k:artifact_job.metadata[k] for k in ['fine_tuned_model', 'model', 'hyperparams']})
    fine_tuned_model = artifact_job.metadata['fine_tuned_model']
    df = pd.read_json(st.VALIDATION_DATA_PATH_JSONL, orient='records', lines=True)

    n_samples = 30
    df = df.iloc[:n_samples]
    data = []

    for _, row in tqdm(df.iterrows()):
        prompt = row['prompt']
        res = openai.Completion.create(model=fine_tuned_model, prompt=prompt, max_tokens=300, stop = ["\n"])
        completion = res['choices'][0]['text']
        completion = completion[1:]       # remove initial space
        prompt = prompt[:-3]              # remove " ->"
        target = row['completion'][1:-1]        # remove initial space and "\n"
        data.append([prompt, target, completion])

    prediction_table = wandb.Table(columns=['prompt', 'target', 'completion'], data=data)
    wandb.log({'predictions': prediction_table})
    wandb.finish() #work out a way to print the run page link


if __name__ == "__main__":
    main()
    #df = pd.read_json(st.VALIDATION_DATA_PATH_JSONL, orient='records', lines=True)
    #print(df)
    #os.system("openai api fine_tunes.cancel -i ft-HyPADg1xUqM0FKSnfugZ4r9f")