import json
import pandas as pd
from tqdm import tqdm
import torch
import os
import numpy as np
import random
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel

with open('./data/sum_data_train_task2.json', 'r') as file:
    json_data = file.read()
    df_train = pd.read_json(json_data)

# Load JSON into DataFrame
with open('./data/sum_data_dev_task2.json', 'r') as file:
    json_data = file.read()
    df_val = pd.read_json(json_data)

joint_dev, labels_dev = list(df_val['data']), list(df_val['label'])
joint_train, labels_train = list(df_train['data']), list(df_train['label'])

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(1)

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
    
    
model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
hs = 1024

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



model = MyModel(model_name, hs)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = Adam(model.parameters(), lr=5e-6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Split train data into train and dev sets
joint_train_train, joint_train_dev, labels_train_train, labels_train_dev = train_test_split(
    joint_train, labels_train, test_size=0.2, random_state=42)

#Tokenize the data.
encoded_train = tokenizer(joint_train_train, return_tensors='pt',
                      truncation_strategy='only_first', add_special_tokens=True, padding=True)
encoded_dev = tokenizer(joint_train_dev, return_tensors='pt',
                      truncation_strategy='only_first', add_special_tokens=True, padding=True)
encoded_test = tokenizer(joint_dev, return_tensors='pt',
                      truncation_strategy='only_first', add_special_tokens=True, padding=True)

# Create train and dev datasets
train_dataset = CtDataset(encoded_train, labels_train_train)
dev_dataset = CtDataset(encoded_dev, labels_train_dev)
test_dataset = CtDataset(encoded_test, labels_dev)

batch_size = 8

# Create train and dev data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_epochs = 8
best_f1 = 0
best_model_path  = "./deberta_sum_epoch8"
log_file = open("./deberta_log_sum_epoch8.txt", "w")

for epoch in range(num_epochs):
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
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss}")

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
    log_file.write(f"Epoch {epoch+1}/{num_epochs}:\n")
    log_file.write(f"Loss: {average_loss:.4f}\n")
    log_file.write(f"Validation - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f} \n")
    log_file.write("\n")

    print(f"Validation - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), best_model_path)  # Save the model with the highest F1 score

# Save the fine-tuned NLI (textual entailment) model.
torch.save(model.state_dict(), "./model-evidence_selection")


# Evaluation
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

