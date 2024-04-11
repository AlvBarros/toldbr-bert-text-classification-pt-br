import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
import pandas as pd
from bert_classifier import BERTClassifier
from text_classification_dataset import TextClassificationDataset

# Load data from file
train_data = pd.read_csv('data/train.csv')
def format_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove words that begin with @
    text = re.sub(r'@\w+', '', text)
    # Remove words that begin with #
    text = re.sub(r'\b#\w+\b', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    pt_stp_words = stopwords.words('portuguese')
    text = ' '.join([word for word in text.split() if word not in pt_stp_words])
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    return text

# Apply the format_text function to the 'text' column in train_data
train_data['text'] = train_data['text'].apply(format_text)
test_data = pd.read_csv('data/test.csv')
features = ['text', 'label']

texts = train_data['text'].tolist()
labels = train_data['label'].tolist()

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up parameters
bert_model_name = 'neuralmind/bert-base-portuguese-cased'
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 2
learning_rate = 2e-5

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load('bert_pt_classifier.pth'))

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

torch.save(model.state_dict(), "bert_pt_classifier.pth")