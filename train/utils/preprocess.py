import pandas as pd
import numpy as np
import emoji
import utils.settings as st
import random
from utils.model import prepare_data, approximate_costs
import csv 

def txt_to_csv(raw_data_path: str, raw_data_processed_path: str):
    """
    Converts the desired txt conversation to csv.
    """
    data = pd.DataFrame(columns = ["prompt", "completion"])
    f = open(raw_data_path, "r", encoding="utf8")
    row = 0

    while f.readline() != "":
        text = f.readline()

        string_searched = "- " + st.USER_NAME + ":"
        name_pos = text.find(string_searched)

        if name_pos != -1:
            text = text[name_pos + len(string_searched):].strip()
            text = process_text(text)
            if text != "":
                data.loc[row] = ["", text]
                row += 1
    
    # Get random sample.
    sample = random.sample(list(range(0, len(data))), min(len(data), st.RANDOM_SAMPLE_SIZE))
    random_data = data.loc[data.index[sample]]
    random_data.reset_index(drop = True, inplace = True)
    random_data.to_csv(raw_data_processed_path)
    return random_data


def txt_to_csv_2(raw_data_path: str, raw_data_processed_path: str):
    """
    Converts the desired txt conversation to csv.
    """
    data = pd.DataFrame(columns = ["prompt", "completion"])
    with open(raw_data_path, "r", encoding="utf8") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    row = 0
    start = True

    prompt_messages = ""
    completion_messages = ""
    last_turn = st.COMPLETION_NAME
    current_turn = st.PROMPT_NAME
    turns = 0

    for text in lines[1:]:

        # While the prompt name not found, continue.
        if text.find(st.PROMPT_NAME) != -1 and start:
            continue
        start = False

        # Get user name in this line and get pos.
        user_name = st.PROMPT_NAME if text.find(st.PROMPT_NAME) != -1 else st.COMPLETION_NAME

        if user_name != current_turn:
            last_turn = current_turn
            current_turn = user_name
            turns += 1
            if last_turn == st.COMPLETION_NAME and current_turn == st.PROMPT_NAME and turns == 2:
                turns = 0
                data.loc[row] = [
                    "", #(emoji.replace_emoji(st.PROMPT_NAME).strip() + ": " + prompt_messages).capitalize(), 
                    (completion_messages).capitalize()
                ]
                row += 1
                prompt_messages = ""
                completion_messages = ""

        string_searched = user_name + ":"
        name_pos = text.find(string_searched)

        # Get message.
        text = text[name_pos + len(string_searched):].strip()
        text = process_text(text)
        if text != "":
            if user_name == st.PROMPT_NAME:
                prompt_messages = join_messages(prompt_messages, text)
            else:
                completion_messages = join_messages(completion_messages, text)

    # Get random sample.
    #sample = random.sample(list(range(0, len(data))), min(len(data), st.RANDOM_SAMPLE_SIZE))
    #random_data = data.loc[data.index[sample]]
    random_data = data.copy()
    random_data.reset_index(drop = True, inplace = True)
    random_data.to_csv(raw_data_processed_path)
    return random_data

def join_messages(text1, text2):
    if len(text1) > 0:
        if text1[-1] != " ":
            return text1 + " " + text2
        return  text1 + text2
    return text2

def process_text(text):
    """
    Remove stop words, urls and emojis.
    """
    # Remove chat stop words.
    for word in st.CHAT_STOP_WORDS:
        if text.find(word) != -1:
            return ""

    # Remove urls.
    if text.find("http") != - 1 or text.find(".com") != -1:
        return ""

    # Remove emojis.
    text = emoji.replace_emoji(text)

    return text

def get_train_and_validation_data(
        raw_data_processed_path: str, 
        train_data_path: str, 
        validation_data_path: str, 
        validation_size = 0.1, 
        save_files = True
    ):

    data = pd.read_csv(raw_data_processed_path)
    data["completion"] = data["completion"].astype(str)
    num_data = len(data)
    print("Total rows:", num_data)

    validation_size = validation_size
    val_amount = int(num_data * validation_size)
    print("Val data:", val_amount)
    train_amount = num_data - val_amount 
    print("Train data:", train_amount)

    #print(data.info())

    approximate_costs(data, validation_size)

    training_data = data.iloc[0: train_amount].copy()
    validation_data = data.iloc[train_amount: ].copy()

    # Save training data.
    if save_files:
        print("Saving files...")
        training_data.to_csv(train_data_path)
        prepare_data(train_data_path)

        # Save validation data.
        validation_data.to_csv(validation_data_path)
        prepare_data(validation_data_path)

if __name__ == "__main__":
    txt_to_csv_2(
        st.RAW_DATA_PATH,
        st.RAW_DATA_PROCESSED_PATH
    )
    """
    get_train_and_validation_data(
        st.RAW_DATA_PROCESSED_PATH, 
        st.TRAIN_DATA_PATH, 
        st.VALIDATION_DATA_PATH, 
        validation_size=0.1, 
        save_files = False
    )
    """