import torch
import torch.nn as nn
import torch.optim as optim
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


class ContextDataset(Dataset):
    def __init__(self, x_inputs, tokenizer, max_len):
        self.x = x_inputs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        text = self.x[index]

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
        }

txt = ["I love", "You love me"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ContextDataset(txt, tokenizer, 5)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

audio_example = torch.randint(0, 10, (1, 8)) # B S

model = Audio_Generator(target_vocab_size = 10, embed_dim = 100, decoder_nhead = 2, decoder_num_layers = 1)
for i, embeddings in enumerate(train_loader):
    print(embeddings)
    print(model(embeddings, audio_example).shape)
    print(model.predict(embeddings))
    break




