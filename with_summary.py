import json
import pandas as pd
import torch
import os
import numpy as np
import random
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import argparse

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):
    # Loading the data
    with open(args.train_path, 'r') as file:
        json_data = file.read()
        df_train = pd.read_json(json_data)

    with open(args.dev_path, 'r') as file:
        json_data = file.read()
        df_val = pd.read_json(json_data)

    joint_dev, labels_dev = list(df_val['data']), list(df_val['label'])
    joint_train, labels_train = list(df_train['data']), list(df_train['label'])

    # Setting the seed
    set_seed(args.seed)

    # Dataset definition (assuming it's defined elsewhere in your code)
   
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

    # Model definition (assuming it's defined elsewhere in your code)
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
            logits = self.classifier(self.dropout(output.last_hidden_state[:, 0, :]))
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return {"logits": logits, "loss": loss}



    # Model instantiation
    model = MyModel(args.model_name, args.hidden_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    optimizer = Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Data preprocessing
    joint_train_train, joint_train_dev, labels_train_train, labels_train_dev = train_test_split(
        joint_train, labels_train, test_size=0.2, random_state=42)

    encoded_train = tokenizer(joint_train_train, return_tensors='pt', truncation_strategy='only_first', add_special_tokens=True, padding=True)
    encoded_dev = tokenizer(joint_train_dev, return_tensors='pt', truncation_strategy='only_first', add_special_tokens=True, padding=True)
    encoded_test = tokenizer(joint_dev, return_tensors='pt', truncation_strategy='only_first', add_special_tokens=True, padding=True)

    train_dataset = CtDataset(encoded_train, labels_train_train)
    dev_dataset = CtDataset(encoded_dev, labels_train_dev)
    test_dataset = CtDataset(encoded_test, labels_dev)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    best_f1 = 0
    best_model_path  = "./deberta_sum_epoch" + str(args.num_epochs)
    log_file = open("./deberta_log_sum_epoch" + str(args.num_epochs) + ".txt", "w")

    # Training and evaluation loop
        # Training and evaluation loop
    for epoch in range(args.num_epochs):
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
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {average_loss}")

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
        log_file.write(f"Epoch {epoch+1}/{args.num_epochs}:\n")
        log_file.write(f"Loss: {average_loss:.4f}\n")
        log_file.write(f"Validation - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}\n")
        log_file.write("\n")

        print(f"Validation - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)

    # Save the final model state
    torch.save(model.state_dict(), args.model_save_path)

    # Final evaluation on test set
    model.eval()
    total_correct = 0
    total_samples = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in test_loader:
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
    print(f"Test - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # File paths
    parser.add_argument('--train_path', type=str, default='./data/sum_data_train_task2.json')
    parser.add_argument('--dev_path', type=str, default='./data/sum_data_dev_task2.json')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    main(args)
