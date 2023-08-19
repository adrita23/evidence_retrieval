import argparse
import os
import json
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModel
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_BOW(section_data, tokenizer):
    document_tokens = tokenizer.tokenize(section_data)
    bow_vector = torch.zeros(50265)  # Initialize BOW vector with zeros
    for token in document_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        bow_vector[token_id] += 1
    normalized_bow_vector = bow_vector / torch.norm(bow_vector)
    global_features = normalized_bow_vector.unsqueeze(0)
    return global_features

def generate_evidence_data(file_path, tokenizer):
    # Read the file.
    df = pd.read_json(file_path)
    df = df.transpose()
    claims = df.Statement.tolist()
    primary_cts, secondary_cts = df.Primary_id, df.Secondary_id
    primary_indices = df.Primary_evidence_index
    secondary_indices = df.Secondary_evidence_index
    sections, types = df.Section_id, df.Type

    primary_evidence_sentences = list()
    secondary_evidence_sentences = list()

    for idx in tqdm(range(len(claims))):
        file_name = "./data/CTs/" + primary_cts[idx] + ".json"
        with open(file_name, 'r') as f:
            data = json.load(f)
            primary_evidence_sentences.append(data[sections[idx]])
        if types[idx] == "Comparison":
            file_name = ".data/CTs/" + secondary_cts[idx] + ".json"
            with open(file_name, 'r') as f:
                data = json.load(f)
                secondary_evidence_sentences.append(data[sections[idx]])
        else:
            secondary_evidence_sentences.append(list())

    joint_data, labels, all_global_context = [], [], []
    for claim_id in range(len(claims)):
        claim = claims[claim_id]
        primary_sents = primary_evidence_sentences[claim_id]
        sec = ' '.join(primary_sents)
        global_context = compute_BOW(sec, tokenizer)
        section_name = sections[claim_id]

        for sid in range(len(primary_sents)):
            candidate_sentence = section_name + " " + primary_sents[sid]
            joint_data.append([candidate_sentence, claim])
            labels.append(sid in primary_indices[claim_id])
            all_global_context.append(global_context)
        if types[claim_id] == "Comparison":
            secondary_sents = secondary_evidence_sentences[claim_id]
            sec = ' '.join(secondary_sents)
            global_context = compute_BOW(sec, tokenizer)

            for sid in range(len(secondary_sents)):
                candidate_sentence = section_name + " " + secondary_sents[sid]
                joint_data.append([candidate_sentence, claim])
                labels.append(sid in secondary_indices[claim_id])
                all_global_context.append(global_context)

        labels = [1 if l else 0 for l in labels]

    return joint_data, all_global_context, labels

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
        cls_representation = output.last_hidden_state[:, 0, :]
        global_features = self.global_fc(global_features)
        global_features = torch.relu(global_features)
        global_features = self.projection(global_features)
        concatenated_features = torch.cat((cls_representation, global_features), dim=1)
        concatenated_features = self.dropout(concatenated_features)
        logits = self.classifier(concatenated_features)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return {"logits": logits, "loss": loss}

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, token_type_ids, global_features, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
            batch["global_features"].to(device),
            batch["labels"].to(device)
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids, global_features, labels)
        loss = outputs["loss"]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_train_loss = total_loss / len(train_loader)
    return average_train_loss

def evaluate_model(model, eval_loader, device):
    model.eval()
    predictions, true_labels = [], []

    for batch in eval_loader:
        input_ids, attention_mask, token_type_ids, global_features, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
            batch["global_features"].to(device),
            batch["labels"].to(device)
        )
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids, global_features)
        logits = outputs["logits"]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        predictions.append(np.argmax(logits, axis=1))
        true_labels.append(label_ids)

    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    return precision, recall, f1, accuracy


def main(args):
    set_seed(args.seed)
    models = [
        "michiyasunaga/BioLinkBERT-base",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    ]
    model_name = args.model_name if args.model_name in models else models[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    joint_train, context_train, labels_train = generate_evidence_data(args.train_path, tokenizer)
    joint_dev, context_dev, labels_dev = generate_evidence_data(args.dev_path, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    model = MyModel(model_name, args.hidden_size, tokenizer.vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    best_f1 = 0

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1}")
        
        # Training
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluation
        precision, recall, f1, accuracy = evaluate_model(model, dev_loader, device)
        print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save the best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pt')
            
    # After all epochs, you can load the best model and test it using test data.
    model.load_state_dict(torch.load('best_model.pt'))
    test_data, test_global, test_labels = generate_evidence_data(args.test_path, tokenizer)
    test_loader = DataLoader(your_dataset_method(test_data, test_global, test_labels), batch_size=args.batch_size)
    precision, recall, f1, accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural model for Evidence Retrieval.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--model_name", type=str, help="Name of the pretrained model.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden layer size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to validation data.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test data.")

    args = parser.parse_args()
    main(args)
