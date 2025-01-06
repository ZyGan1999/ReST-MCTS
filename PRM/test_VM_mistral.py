import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd

max_length = 900

# Load the pre-trained Mistral-7b model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/ganzeyu/Mistral-7B-MetaMath", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("/data/ganzeyu/Mistral-7B-MetaMath", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
#tokenizer.pad_token = tokenizer.eos_token

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, data_js, tokenizer):
        self.data_js = data_js
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        prompt_answer = self.data_js[idx]['prompt_answer']
        label = self.data_js[idx]['label']

        encoded_pair = self.tokenizer.encode_plus(
            prompt_answer,
            padding='max_length',
            max_length=max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
            'label': label
        }


class Mistral_VM(nn.Module):
    def __init__(self, base, vocab_size=32000):
        super(Mistral_VM, self).__init__()
        # self.base_model = base
        # self.LN = nn.Linear(vocab_size, 1)
        # 1) 冻结 base_model 的所有参数
        for param in base.parameters():
            param.requires_grad = False

        self.base_model = base
        # 2) 仅保留 LN 这一层的可训练参数
        self.LN = nn.Linear(vocab_size, 1)


    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)


# Load training set, validation set, and test set data
#train_js = 'data/train_en.json'
test_js = 'data/test_en.json'
val_js = 'data/valid_en.json'


def read_json(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


#train_json = read_json(train_js)  # This section uses a CSV file as an example to describe how to load data
val_json = read_json(val_js)
test_json = read_json(test_js)

# Create a custom dataset
#train_dataset = MyDataset(train_json, tokenizer)
val_dataset = MyDataset(val_json, tokenizer)
test_dataset = MyDataset(test_json, tokenizer)

# Create data loaders
batch_size = 3  # Set batch size
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Set device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, '\n')
vocab_size = base_model.config.vocab_size
print(vocab_size)

# Load the best model for inference
best_model = Mistral_VM(base_model, vocab_size)
best_model.load_state_dict(torch.load("records/Mistral/VM_best_checkpoint.pt"))
#checkpoint = torch.load("records/Mistral/VM_best_checkpoint.pt", map_location="cpu")
#best_model.LN.load_state_dict(checkpoint)
best_model.to(device)
best_model.eval()

# Perform inference
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(dtype=torch.float32).to(device)
        outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
        test_preds.extend(outputs.tolist())
        test_labels.extend(labels.tolist())
    print("Inference results:")
    for i in range(len(test_preds)):
        print(f"Sample {i + 1}: Predicted score {test_preds[i]}, Actual score {test_labels[i]}, Truncated score {min(max(test_preds[i], 0), 1)}")

cnt = 0
for i in range(len(test_preds)):
    if abs(min(max(test_preds[i], 0), 1) - test_labels[i]) <= 0.1:
        cnt += 1
test_acc = cnt / len(test_preds)
print(f"Test accuracy: {test_acc:.4f}")
