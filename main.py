import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import gensim
import nltk
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import Trainer, TrainingArguments,BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from transformers import AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
from sklearn.metrics import f1_score
from model import Audio_Generator
from utils import *


class ContextDataset(Dataset):
    def __init__(self, x_inputs, tokenizer, max_len, audios):
        self.x = x_inputs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.audios = audios

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        text = self.x[index]
        audio = self.audios[index]

        embedding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            "text": text,
            "input_ids": embedding['input_ids'].flatten(),
            "attention_mask": embedding['attention_mask'].flatten(),
            "token_type_ids": embedding['token_type_ids'].flatten()
        }, audio



parser = argparse.ArgumentParser()
# experimental settings
parser.add_argument("--base_path",
    type=str,
    default="./maestro-v2.0.0/",
    help="base_path of audio data")
parser.add_argument("--csv_path", 
    type=str,
    default="./data/clean+GPTcaption.csv",
    help="csv dataset path")
parser.add_argument("--max_text_lenth", 
    type=int,
    default=512,
    help="max_lenth for bert tokenizer")
parser.add_argument("--batch_size", 
    type=int,
    default=5,
    help="batch size for dataloader")
parser.add_argument("--frame", 
    type=int,
    default=100,
    help="#frame/sec for audio")
args = parser.parse_args()

base_path = args['base_path'] # './maestro-v2.0.0/'
csv_path = args['csv_path'] # './data/clean+GPTcaption.csv'

data = read_csv(csv_path)

midi_files = get_midi_files(data,base_path)

text_descriptions = get_text_descriptions(data)

num_data = len(text_descriptions)
train_num = int(num_data * 0.8)
valid_num = int((num_data-train_num)/2)
test_num = num_data - train_num - valid_num

train_text = text_descriptions[:train_num] # [train_num]
valid_text = text_descriptions[train_num:train_num+valid_num]
test_text = text_descriptions[train_num+valid_num:]

audios = midi_files_to_audios(midi_files,args['frame'])
train_audios = audios[:train_num] # [train_num, frame*time]
valid_audios = audios[train_num:train_num+valid_num]
test_audios = audios[train_num+valid_num:]

# txt = ["I love", "You love me"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ContextDataset(train_text, tokenizer, args['max_text_lenth'] , train_audios)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= args['batch_size'], shuffle=True)

valid_dataset = ContextDataset(valid_text, tokenizer, args['max_text_lenth'] , valid_audios)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size= args['batch_size'], shuffle=False)

test_dataset = ContextDataset(test_text, tokenizer, args['max_text_lenth'] , test_audios)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= args['batch_size'], shuffle=False)

# audio_example = torch.randint(0, 10, (1, 8)) # B S

model = Audio_Generator(target_vocab_size = 10, embed_dim = 100, decoder_nhead = 2, decoder_num_layers = 1)
for i, batchi in enumerate(train_loader):
    embeddings, audio = batchi
    print(embeddings)
    print(model(embeddings, audio).shape)
    print(model.predict(embeddings))
    break




