import json
import pandas as pd
from tqdm import tqdm

# Constants
TRAIN_PATH = "./data/train.json"
DEV_PATH = "./data/dev.json"
TEST_PATH = "./data/test.json"


def generate_evidence_data(file_path, path_type):
    """
    Generates data from clinical trials for Task 2: Evidence Retrieval (/selection).

    Args:
    - file_path (str): Path to the JSON of the dataset.
    - path_type (str): Type of data (training or validation).

    Returns:
    - joint_data: List of training instances in form of "claim [SEP] candidate_sentence"
    - labels: List of labels, 0 if candidate_sentence is not evidence, 1 if it is
    """
    df = pd.read_json(file_path).transpose()

    # Extract required columns
    claims, primary_cts, secondary_cts = df.Statement.tolist(), df.Primary_id, df.Secondary_id
    primary_indices, secondary_indices, sections, types = df.Primary_evidence_index, df.Secondary_evidence_index, df.Section_id, df.Type

    primary_evidence_sentences, secondary_evidence_sentences = [], []
    primary_evidence_summary, secondary_evidence_summary = [], []

    # Extract evidence data
    for idx in tqdm(range(len(claims))):
        file_name = f"./data/CTs/{primary_cts[idx]}.json"

        with open(file_name, 'r') as f:
            data = json.load(f)
            primary_evidence_sentences.append(data[sections[idx]])

        summary_path = '.data/sum_train_primary' if path_type == 'tr' else '.data/sum_dev_primary'
        with open(f'{summary_path}/{idx}.txt', 'r') as f:
            primary_evidence_summary.append(f.read())

        if types[idx] == "Comparison":
            file_name = f".data/CTRs/{secondary_cts[idx]}.json"

            with open(file_name, 'r') as f:
                data = json.load(f)
                secondary_evidence_sentences.append(data[sections[idx]])

            summary_path = '.data/sum_train_secondary' if path_type == 'tr' else '.data/sum_dev_secondary'
            with open(f'{summary_path}/{idx}.txt', 'r') as f:
                secondary_evidence_summary.append(f.read())
        else:
            secondary_evidence_sentences.append([])
            secondary_evidence_summary.append([])

    # Generate training instances in form of "claim [SEP] candidate_sentence"
    joint_data, labels = [], []

    for claim_id in range(len(claims)):
        claim, primary_sents, primary_sum, section_name = claims[claim_id], primary_evidence_sentences[claim_id], primary_evidence_summary[claim_id], sections[claim_id]

        for sid, sentence in enumerate(primary_sents):
            candidate_sentence = f"{section_name} {sentence}"
            joint_data.append(f"{candidate_sentence} [SEP] {claim} [SEP] {primary_sum}")
            labels.append(sid in primary_indices[claim_id])

        if types[claim_id] == "Comparison":
            for sid, sentence in enumerate(secondary_evidence_sentences[claim_id]):
                candidate_sentence = f"{section_name} {sentence}"
                joint_data.append(f"{candidate_sentence} [SEP] {claim} [SEP] {secondary_evidence_summary[claim_id]}")
                labels.append(sid in secondary_indices[claim_id])

    labels = [1 if label else 0 for label in labels]

    return joint_data, labels


def save_to_json(data, path):
    """
    Save data to a JSON file.
    
    Args:
    - data (pd.DataFrame): DataFrame containing data to be saved.
    - path (str): Path where JSON should be saved.
    """
    with open(path, 'w') as file:
        file.write(data.to_json(orient='records'))


if __name__ == "__main__":
    # Generate evidence data
    df_train = generate_evidence_data(TRAIN_PATH, 'tr')
    df_val = generate_evidence_data(DEV_PATH, 'vl')
    dt_tr, lb_tr = df_train[0], df_train[1]
    dt_vl, lb_vl = df_val[0], df_val[1]

    # Create DataFrames
    df_train = pd.DataFrame({'data': dt_tr, 'label': lb_tr})
    df_val = pd.DataFrame({'data': dt_vl, 'label': lb_vl})

    # Save to JSON
    save_to_json(df_train, '.data/sum_data_train_task2.json')
    save_to_json(df_val, '.data/sum_data_dev_task2.json')
