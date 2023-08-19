import numpy as np
import json
import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

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

models = ["michiyasunaga/BioLinkBERT-base",
"MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
]
model_name = models[0]
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def compute_BOW(section_data):
    document_tokens = tokenizer.tokenize(section_data)
    bow_vector = torch.zeros(50265)  # Initialize BOW vector with zeros
    for token in document_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        bow_vector[token_id] += 1

    # Normalize the BOW vector
    normalized_bow_vector = bow_vector / torch.norm(bow_vector)
    global_features = normalized_bow_vector.unsqueeze(0)
    return global_features

TRAIN_PATH = "./data/train.json"
DEV_PATH = "./data/dev.json"
TEST_PATH = "./data/test.json"

def generate_evidence_data(file_path):
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

    for idx in tqdm(range(len(claims))):
        file_name = "./data/CTs/" + primary_cts[idx] + ".json"

        #Create a list of all evidence sentences from the primary trial for this claim.
        with open(file_name, 'r') as f:
            data = json.load(f)
            primary_evidence_sentences.append(data[sections[idx]])

        #If it is a comparative claim, also create a list of secondary-trial evidence sentences.
        if types[idx] == "Comparison":
            # file_name = "/home/ubuntu/nli4ct/Complete_dataset/CTs/" + secondary_cts[idx] + ".json"
            file_name = ".data/CTs/" + secondary_cts[idx] + ".json"

            with open(file_name, 'r') as f:
                data = json.load(f)
                secondary_evidence_sentences.append(data[sections[idx]])
        else:
            secondary_evidence_sentences.append(list())

    #Generate training instances in form of "claim [SEP] candidate_sentence",
    joint_data = list()

    #Label is 0 if candidate sentece is not evidence for this claim, 1 if it is
    labels = list()
    all_global_context = list()
    for claim_id in range(len(claims)):
        claim = claims[claim_id]
        primary_sents = primary_evidence_sentences[claim_id]
        sec = ' '.join(primary_sents)
        global_context = compute_BOW(sec)
        section_name = sections[claim_id]

        for sid in range(len(primary_sents)):
            candidate_sentence = section_name + " " + primary_sents[sid]
            # j = candidate_sentence + " [SEP] " + claim
            j = [candidate_sentence, claim]
            joint_data.append(j)
            labels.append(sid in primary_indices[claim_id])
            all_global_context.append(global_context)

        if types[claim_id] == "Comparison":
            secondary_sents = secondary_evidence_sentences[claim_id]
            sec = ' '.join(secondary_sents)
            global_context = compute_BOW(sec)


            for sid in range(len(secondary_sents)):
                candidate_sentence = section_name + " " + secondary_sents[sid]
                # j = candidate_sentence + " [SEP] " + claim
                j = [candidate_sentence, claim]
                joint_data.append(j)
                labels.append(sid in secondary_indices[claim_id])
                all_global_context.append(global_context)

        labels = [1 if l else 0 for l in labels]

    return joint_data, all_global_context, labels


joint_train, context_train, labels_train = generate_evidence_data(TRAIN_PATH)
joint_dev, context_dev, labels_dev = generate_evidence_data(DEV_PATH)


hs = 768

class MyModel(nn.Module):
    def __init__(self, model_name, hidden_size, vocab_size):
        super(MyModel, self).__init__()
        self.emb_layer = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.global_fc = nn.Linear(self.vocab_size, self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size*2, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, global_features, labels=None):

        output = self.emb_layer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print("emb", output.shape)
        cls_representation = output.last_hidden_state[:, 0, :]  # CLS token representation
        # print("cls", cls_representation.shape)
        global_features = self.global_fc(global_features)
        global_features = torch.relu(global_features)
        global_features = self.projection(global_features)
        # print("gf", global_features.shape)
        concatenated_features = torch.cat((cls_representation, global_features), dim=1)
        concatenated_features = self.dropout(concatenated_features)  # Apply dropout
        # print("cat", concatenated_features.shape)
        logits = self.classifier(concatenated_features)

        # print(output['last_hidden_state'].shape)
        # logits = self.classifier(self.dropout(output['pooler_output']))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {"logits": logits, "loss": loss}



class CtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, context, labels):
        self.encodings = encodings
        self.context = context
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['context'] = self.context[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size
model = MyModel(model_name, hs, vocab_size)

optimizer = Adam(model.parameters(), lr=5e-6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Split train data into train and dev sets
joint_train_train, joint_train_dev, labels_train_train, labels_train_dev, context_train_train, context_train_dev = train_test_split(
    joint_train, labels_train, context_train, test_size=0.2, random_state=42)

#Tokenize the data.
encoded_train = tokenizer(joint_train_train, return_tensors='pt',
                      truncation_strategy='only_first', add_special_tokens=True, padding=True)
encoded_dev = tokenizer(joint_train_dev, return_tensors='pt',
                      truncation_strategy='only_first', add_special_tokens=True, padding=True)
encoded_test = tokenizer(joint_dev, return_tensors='pt',
                      truncation_strategy='only_first', add_special_tokens=True, padding=True)



train_dataset = CtDataset(encoded_train, context_train_train, labels_train_train)
dev_dataset = CtDataset(encoded_dev, context_train_dev, labels_train_dev)
test_dataset = CtDataset(encoded_test, context_dev, labels_dev)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


num_epochs = 8
best_f1 = 0
best_model_path  = "./seed1/bow_epoch8"
log_file = open("./seed1/bow_raw_epoch8.txt", "w")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        context = batch['context'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, token_type_ids, context, labels=labels)
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
    log_file.write(f"Validation - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}\n")
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


log_file.write(f"Epoch {epoch+1}/{num_epochs}:\n")
log_file.write(f"Loss: {average_loss:.4f}\n")
log_file.write(f"Test Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}\n")
log_file.write("\n")

print(f"Test - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")
