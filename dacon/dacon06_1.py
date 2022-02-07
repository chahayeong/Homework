import pandas as pd
from glob import glob
import os
import numpy as np
from tqdm import tqdm, tqdm_notebook


import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW

train = pd.read_csv("./dacon/_data06/train_data.csv")
test = pd.read_csv("./dacon/_data06/test_data.csv")
submission = pd.read_csv("./dacon/_data06/sample_submission.csv")

max_len = 70
batch_size = 128
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

device = torch.device("cuda:0")

bertmodel, vocab = get_pytorch_kobert_model(cachedir = ".cache")

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                pad, pair, mode = "train"):
        self.mode = mode
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length = max_len, pad = pad, pair = pair)
        if self.mode == "train":
            self.sentences = [transform([i[sent_idx]]) for i in dataset]
            self.labels = [np.int32(i[label_idx]) for i in dataset]
            
        else:
            self.sentences = [transform(i) for i in dataset]
        
    def __getitem__(self, i):
        if self.mode == 'train':
            return (self.sentences[i] + (self.labels[i], ))
        else:
            return self.sentences[i]
    
    def __len__(self):
        return (len(self.sentences))
    
print(pd.unique(train["label"]))

label_dict = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}

train["premise_"] = "[CLS]" + train["premise"] + "[SEP]"
train["hypothesis_"] = train["hypothesis"] + "[SEP]"

test["premise_"] = "[CLS]" + test["premise"] + "[SEP]"
test["hypothesis_"] = test["hypothesis"] + "[SEP]"

train["text_sum"] = train.premise_ + " " + train.hypothesis_
test["text_sum"] = test.premise_ + " " + test.hypothesis_

train_content = []
test_content = []

for i, text in enumerate(train.text_sum):
    train_content.append(list([text, str(label_dict[train.label[i]])]))
    
for i, text in enumerate(test.text_sum):
    test_content.append([text])
    
dataset_train = train_content[:20000]
dataset_valid = train_content[20000:]
dataset_test = test_content

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False, mode = "train")
data_valid = BERTDataset(dataset_valid, 0, 1, tok, max_len, True, False, mode = "train")
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False, mode = "test")

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, num_workers = 5)
valid_dataloader = torch.utils.data.DataLoader(data_valid, batch_size = batch_size, num_workers = 5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, num_workers = 5)

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size = 768, num_classes=3, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
model = BERTClassifier(bertmodel, dr_rate = 0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    valid_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_acc += calc_accuracy(out, label)

    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        valid_acc += calc_accuracy(out, label)
    print("epoch {} valid acc {}".format(e+1, valid_acc / (batch_id+1)))
    
result = []
model.eval()
with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        result.append(model(token_ids, valid_length, segment_ids))
        
        
result_ = []
for i in result:
    for j in i:
        result_.append(int(torch.argmax(j)))
        
out = [list(label_dict.keys())[_] for _ in result_]

submission["label"] = out

submission.to_csv("sample_submission.csv", index = False)

submission.sample(3)
    