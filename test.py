import torch
from torch import nn
import pandas as pd
from transformers import BertTokenizer
from bert_classifier import BERTClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the .pth file
model = BERTClassifier('neuralmind/bert-base-portuguese-cased', 2)
model.load_state_dict(torch.load('bert_pt_classifier.pth'))
model.eval()
if torch.cuda.is_available():
      model.cuda()

# Perform inference or further processing using the loaded model

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
    return 1 if preds.item() == 1 else 0

# Apply predict_sentiment function to each row in the test_data DataFrame
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

print(predict_sentiment('Hello World!', model, tokenizer, device)) # 0