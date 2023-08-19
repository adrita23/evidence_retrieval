import os
import random
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, Trainer, TrainingArguments

TRAIN_PATH = "./data/train.json"
DEV_PATH = "./data/dev.json"
TEST_PATH = "./data/test.json"


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_evidence_data(file_path, path_type):
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

    primary_evidence_summary = list()
    secondary_evidence_summary = list()

    for idx in tqdm(range(len(claims))):
        if path_type == 'tr':
            with open(f'.data/sum_train_primary/{idx}.txt', 'r') as f:
              content = f.read()
              primary_evidence_summary.append(content)

        else:
            with open(f'.data/sum_dev_primary/{idx}.txt', 'r') as f:
              content = f.read()
              primary_evidence_summary.append(content)

        #If it is a comparative claim, also create a list of secondary-trial evidence sentences.
        if types[idx] == "Comparison":
            if path_type == 'tr':
                with open(f'.data/sum_train_secondary/{idx}.txt', 'r') as f:
                  content = f.read()
                  secondary_evidence_summary.append(content)

            else:
                with open(f'./data/sum_dev_secondary/{idx}.txt', 'r') as f:
                  content = f.read()
                  secondary_evidence_summary.append(content)

    if path_type == "tr":
        filename1 = '/content/drive/MyDrive/NLI4CT_3rd/data/tr_primary_evidence_sentences.json'
        filename2 = '/content/drive/MyDrive/NLI4CT_3rd/data/tr_secondary_evidence_sentences.json'

    elif path_type == "vl":
        filename1 = '/content/drive/MyDrive/NLI4CT_3rd/data/val_primary_evidence_sentences.json'
        filename2 = '/content/drive/MyDrive/NLI4CT_3rd/data/val_secondary_evidence_sentences.json'

    # # Load list1
    with open(filename1, 'r') as file:
        primary_evidence_sentences = json.load(file)

    # Load list2
    with open(filename2, 'r') as file:
        secondary_evidence_sentences = json.load(file)

    # Generate training instances in form of "claim [SEP] candidate_sentence",
    joint_data = list()

    #Label is 0 if candidate sentece is not evidence for this claim, 1 if it is
    labels = list()


    for claim_id in range(len(claims)):
        claim = claims[claim_id]
        primary_sents = primary_evidence_sentences[claim_id]
        primary_sum  = primary_evidence_summary[claim_id]


        for sid in range(len(primary_sents)):
            candidate_sentence = primary_sents[sid]
            # j = candidate_sentence + " [SEP] " + claim + " [SEP] " + primary_sum
            j = [candidate_sentence, claim , primary_sum]
            joint_data.append(j)
            labels.append(sid in primary_indices[claim_id])

        if types[claim_id] == "Comparison":
            secondary_sents = secondary_evidence_sentences[claim_id]
            secondary_sum  = secondary_evidence_summary[claim_id]

            for sid in range(len(secondary_sents)):
                candidate_sentence = secondary_sents[sid]
                # j = candidate_sentence + " [SEP] " + claim + " [SEP] " + secondary_sum
                j = [candidate_sentence,  claim , secondary_sum]
                joint_data.append(j)
                labels.append(sid in secondary_indices[claim_id])

        labels = [1 if l else 0 for l in labels]

    return joint_data, labels

df_train = generate_evidence_data(TRAIN_PATH, 'tr')
df_val = generate_evidence_data(DEV_PATH, 'vl')

dt_tr, lb_tr = df_train[0], df_train[1]
dt_vl, lb_vl = df_val[0], df_val[1]

class MyModel(nn.Module):
    def __init__(self, model_name, hidden_size):
        super(MyModel, self).__init__()
        self.emb_layer = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.classifier1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.classifier2 = nn.Linear(hidden_size * 2, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bilstm = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True, dropout = 0.2)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.emb_layer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        # print(last_hidden_state.shape)
        batch_size = input_ids.size(0)

        sent1_indices = []
        sent2_indices = []
        sent1_representations = []
        sent2_representations = []

        for i in range(batch_size):
            sep_indices = (input_ids[i] == self.tokenizer.sep_token_id).nonzero()
            s1 = outputs.last_hidden_state[i, 1:sep_indices[1][0], : ]
            s2 = outputs.last_hidden_state[i, sep_indices[1][0]:sep_indices[2][0], : ]
            # Perform max pooling on s1 and s2
            s1_pooled = torch.max(s1, dim=0, keepdim=True)[0]
            s2_pooled = torch.max(s2, dim=0, keepdim=True)[0]

            sent1_representations.append(s1_pooled)
            sent2_representations.append(s2_pooled)

        # Concatenate s1 and s2 representations
        sent1_representations = torch.cat(sent1_representations, dim=0)
        sent2_representations = torch.cat(sent2_representations, dim=0)
        # print("sen", sent1_representations.shape, sent2_representations.shape)
        # Stack the batch inputs after concatenation
        combined_representations = torch.cat([sent1_representations, sent2_representations], dim=1)
        # print(combined_representations.shape)
        lstm_output, _ = self.bilstm(combined_representations.unsqueeze(0))
        lstm_output = lstm_output.squeeze(0)
        # print("comb", combined_representations.shape)
        # Apply classifier layers
        output1 = self.classifier1(self.dropout(lstm_output))
        output2 = self.classifier2(output1)

        logits = output2.squeeze(1)

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
    model = MyModel(args.model_name, args.hidden_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate joint claim+[SEP]+candidate_sentence data
    joint_train, labels_train = df_train
    joint_dev, labels_dev = df_val

    if not joint_train or not joint_dev or len(joint_train) != len(labels_train) or len(joint_dev) != len(labels_dev):
        raise ValueError("Invalid joint_train or joint_dev data.")

    #Tokenize the data.
    encoded_train = tokenizer(joint_train, return_tensors='pt',
                        truncation_strategy='only_first', add_special_tokens=True, padding=True)
    encoded_dev = tokenizer(joint_dev, return_tensors='pt',
                        truncation_strategy='only_first', add_special_tokens=True, padding=True)



    train_dataset = CtDataset(encoded_train, labels_train)
    dev_dataset = CtDataset(encoded_dev, labels_dev)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, token_type_ids, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {average_loss}")

        # Evaluation
        model.eval()
        total_correct = 0
        total_samples = 0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                logits = outputs['logits']

                _, preds = torch.max(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                true_labels.extend(labels.tolist())
                predicted_labels.extend(preds.tolist())

        accuracy = total_correct / total_samples
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        print(f"Validation - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")

    # Save the fine-tuned NLI (textual entailment) model.
    torch.save(model.state_dict(), args.path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for evidence selection.")
    
    parser.add_argument("--epoch", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the pre-trained model.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--path", type=str, default="./model-evidence_selection", help="Path to save the trained model.")
    
    args = parser.parse_args()

    main(args)