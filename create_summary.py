import json
import os
import random
import multiprocessing
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Constants
TRAIN_PATH = "./data/train.json"
DEV_PATH = "./data/dev.json"
TEST_PATH = "./test.json"
MODEL_NAME = "t5-base"
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)
MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def summarizer1(document, typ):
    """
    Generates a summary for the given document.
    
    Args:
    - document (tuple): Contains document ID and text.
    - typ (str): Type of the document, either 'single' or 'comp'.
    
    Returns:
    - str: Generated summary.
    """
    inputs = TOKENIZER.encode(document[1], return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    summary_ids = MODEL.generate(inputs, num_beams=4, max_length=212, early_stopping=True)
    summary = TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
    
    if typ == 'single':
        folder_name = sum_train_primary
    elif typ == 'comp':
        folder_name = sum_train_secondary

    with open(f'./{folder_name}/{document[0]}.txt', 'w') as f:
        f.write(summary)
    
    return summary


def generate_summary_data(file_path):
    """
    Generates data summaries from clinical trials.
    
    Args:
    - file_path (str): Path to the dataset JSON file.
    
    Returns:
    - list: Evidence sentences.
    """
    df = pd.read_json(file_path).transpose()
    claims = df.Statement.tolist()

    primary_cts, secondary_cts = df.Primary_id, df.Secondary_id
    sections, types = df.Section_id, df.Type
    evidence_summary = []

    for idx in tqdm(range(len(claims))):
        if types[idx] == "Single":
            file_name = f"./data/CTs/{primary_cts[idx]}.json"
            with open(file_name, 'r') as f:
                data = json.load(f)
                evidence_summary.append(summarizer1((' '.join(data[sections[idx]]), 'single')))
                
        if types[idx] == "Comparison":
            file_name = f"./data/CTs/{secondary_cts[idx]}.json"
            with open(file_name, 'r') as f:
                data = json.load(f)
                evidence_summary.append(summarizer1((' '.join(data[sections[idx]]), 'comp')))

    return evidence_summary


def process_data(file_path):
    """
    Process data using multiprocessing.
    
    Args:
    - file_path (str): Path to the dataset JSON file.
    """
    data = generate_summary_data(file_path)
    
    with multiprocessing.Pool() as pool:
        pool.map(summarizer1, data)


if __name__ == "__main__":
    process_data(TRAIN_PATH)
    process_data(DEV_PATH)
