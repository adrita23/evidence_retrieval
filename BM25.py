import argparse
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel
import numpy as np
from rank_bm25 import BM25Okapi
import json
import pandas as pd
from tqdm import tqdm
import os
import random
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_bm25_scores(sents, claim):
    tokenized = [x.split(" ") for x in sents]
    tokenized = [[x.strip(' ') for x in y] for y in tokenized]
    tokenized = [[x for x in y if x] for y in tokenized]
    bm25 = BM25Okapi(tokenized)
    #retrieve and tokenize the statement
    statement = claim.split(" ")
    scores = bm25.get_scores(statement)
    min_val, max_val = min(scores), max(scores)
    normalized_scores = [str((x - min_val) *100/ (max_val - min_val)) for x in scores]
    return normalized_scores


def generate_evidence_data(file_path, type_f):
    '''
    Generates data from clinical trials for Task 2: Evidence Retrieval (/selection).

    Parameters:
        file_path (str): Path to the JSON of the dataset.

    Returns:
        joint_data: List of training instances in form of "claim [SEP] candidate_sentence" (str)
        labels: List of labels, 0 if candidate_sentence is not evidence, 1 if it is
    '''

    #Read the file.
    df = pd.read_json(file_path)
    df = df.transpose()

    #Extract claims.
    claims = df.Statement.tolist()

    #(Prepare to) Extract all evidence sentences from clinical trials
    primary_cts, secondary_cts = df.Primary_id, df.Secondary_id
    primary_indices = df.Primary_evidence_index
    secondary_indices = df.Secondary_evidence_index
    sections, types = df.Section_id, df.Type

    primary_evidence_sentences = list()
    secondary_evidence_sentences = list()

    if type_f == "tr":
        # Save lists as JSON files
        filename1 = './data/tr_primary_evidence_sentences.json'
        filename2 = './data/tr_secondary_evidence_sentences.json'


    elif type_f == "val":
        # Save lists as JSON files
        filename1 = './data/val_primary_evidence_sentences.json'
        filename2 = './data/val_secondary_evidence_sentences.json'


    # # Load list1
    with open(filename1, 'r') as file:
        primary_evidence_sentences = json.load(file)

    # Load list2
    with open(filename2, 'r') as file:
        secondary_evidence_sentences = json.load(file)

    #Generate training instances in form of "claim [SEP] candidate_sentence",
    joint_data = list()

    #Label is 0 if candidate sentece is not evidence for this claim, 1 if it is
    labels = list()

    for claim_id in range(len(claims)):
        claim = claims[claim_id]
        primary_sents = primary_evidence_sentences[claim_id]
        bm25_scores = get_bm25_scores(primary_sents, claim)
        section_name = sections[claim_id]

        for sid in range(len(primary_sents)):
            candidate_sentence = section_name + " " + primary_sents[sid]
            bm25_pts = bm25_scores[sid]
            j = candidate_sentence + " [SEP] " + bm25_pts + " [SEP] " + claim
            # j = [candidate_sentence, bm25_pts, claim]
            joint_data.append(j)
            labels.append(sid in primary_indices[claim_id])

        if types[claim_id] == "Comparison":
            secondary_sents = secondary_evidence_sentences[claim_id]
            bm25_scores = get_bm25_scores(secondary_sents, claim)


            for sid in range(len(secondary_sents)):
                candidate_sentence = section_name + " " + secondary_sents[sid]
                bm25_pts = bm25_scores[sid]
                j = candidate_sentence + " [SEP] " + bm25_pts + " [SEP] " + claim
                # j = [candidate_sentence, bm25_pts, claim]
                joint_data.append(j)
                labels.append(sid in secondary_indices[claim_id])

        labels = [1 if l else 0 for l in labels]

    return joint_data, labels


class MyModel(nn.Module):
    def __init__(self, model_name, hidden_size):
        super(MyModel, self).__init__()
        self.emb_layer = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.emb_layer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(output['last_hidden_state'].shape)
        logits = self.classifier(self.dropout(output.last_hidden_state[:,0,:]))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return {"logits": logits, "loss": loss}

class CtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main(args):
    set_seed(args.seed)

    # Load data
    TRAIN_PATH = "./data/train.json"
    DEV_PATH = "./data/dev.json"
    TEST_PATH = "./data/test.json"

    joint_train, labels_train = generate_evidence_data(TRAIN_PATH, "tr")
    joint_dev, labels_dev = generate_evidence_data(DEV_PATH, "val")

    # Define models
    models = ["michiyasunaga/BioLinkBERT-base",
              "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]

    model_name = args.model_name if args.model_name else models[-1]
    hs = args.hidden_size

    # Initialize model and tokenizer
    model = MyModel(model_name, hs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Encode and create dataloaders
    train_encodings = tokenizer(joint_train, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(joint_dev, truncation=True, padding=True, max_length=512)

    train_dataset = CtDataset(train_encodings, labels_train)
    val_dataset = CtDataset(val_encodings, labels_dev)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_epochs = args.epoch
    best_f1 = 0
    best_model_path = args.path
    log_file = open(f"{best_model_path}.txt", "w")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        log_file.write(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss}\n")

        # Validation
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                input_ids, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)

        log_file.write(f"Validation - Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Accuracy: {accuracy}\n")

        # Save the best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=8, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--model_name", type=str, help="Model name from HuggingFace Model Hub.")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size of the model.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--path", type=str, default="./seed1/deberta_bm25_epoch10", help="Path for saving the best model.")
    args = parser.parse_args()

    main(args)


