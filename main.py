import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import nltk
import time
import os
from parse import parse_args
from torch.optim import lr_scheduler
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
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import accuracy_score  
from sklearn.metrics import f1_score

class ContextDataset(Dataset):
    def __init__(self, x_inputs, tokenizer, max_len, audios, device):
        self.x = x_inputs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.audios = audios
        self.device = device

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

        return {"input_ids": embedding['input_ids'].flatten().to(self.device),
            "attention_mask": embedding['attention_mask'].flatten().to(self.device),
            "token_type_ids": embedding['token_type_ids'].flatten().to(self.device)
        }, audio




args = parse_args()

base_path = args.base_path # './maestro-v2.0.0/'
csv_path = args.data_path + '/clean+GPTcaption.csv' # './data/clean+GPTcaption.csv'
processed_path = args.data_path + '/frame{}_data.npy'.format(args.frame)

writer = SummaryWriter('./tensorboard/log')


if not os.path.exists(processed_path):
    print('processing data...')
    data = read_csv(csv_path)
    midi_files = get_midi_files(data,base_path)
    text_descriptions = get_text_descriptions(data)
    audios = midi_files_to_audios(midi_files, args.frame)
    data_dict = {'text_descriptions':text_descriptions, 'audios':audios}
    np.save(processed_path, data_dict)
else:
    print('loading data...')
    data_dict = np.load(processed_path, allow_pickle=True).item()
    text_descriptions, audios = data_dict['text_descriptions'], data_dict['audios']

num_data = len(text_descriptions)
train_num = int(num_data * 0.8)
valid_num = int((num_data-train_num)/2)
test_num = num_data - train_num - valid_num

train_text = text_descriptions[:train_num] # [train_num]
valid_text = text_descriptions[train_num:train_num+valid_num]
test_text = text_descriptions[train_num+valid_num:]

train_audios = audios[:train_num] # [train_num, frame*time]
valid_audios = audios[train_num:train_num+valid_num]
test_audios = audios[train_num+valid_num:]

print('generating dataset...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# txt = ["I love", "You love me"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ContextDataset(train_text, tokenizer, args.max_text_lenth, train_audios, device)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)

valid_dataset = ContextDataset(valid_text, tokenizer, args.max_text_lenth, valid_audios, device)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size= args.batch_size, shuffle=False)

test_dataset = ContextDataset(test_text, tokenizer, args.max_text_lenth, test_audios, device)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= args.batch_size, shuffle=False)

# audio_example = torch.randint(0, 10, (1, 8)) # B S

model = Audio_Generator(target_vocab_size = 10, embed_dim = 100, decoder_nhead = 2, decoder_num_layers = 1)

model.to(device)
lrlast = .0001
lrmain = .00001
optim1 = torch.optim.Adam(
    [
        {"params":model.text_encoder.bert.parameters(),"lr": lrmain},
        {"params":model.text_encoder.mlp.parameters(),"lr": lrlast},
        {"params":model.audio_generator.parameters(), "lr": lrlast},

   ])
optimizer = optim1
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print('starting')
best_epoch = 0
best_loss = float('inf')
best_acc = 0
metrics_update = 0
for epoch in range(args.num_epochs):
    model.train()
    start_time = time.time()
    loss = 0

    for i, batchi in enumerate(train_loader):
        embeddings, audio = batchi
        audio = audio.to(device)
        model.zero_grad()
        prediction = model(embeddings, audio)
        loss += criterion(prediction, audio)
        loss.backward()
        optimizer.step()
    print('epoch:', epoch)
    writer.add_scalar('train_loss', loss/train_num, epoch)
    print('train_loss {:.4f}'.format(loss/train_num) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    scheduler.step()
    with torch.no_grad():
        all_prediction = None
        all_audio = None
        for i, batchi in enumerate(valid_loader):
            embeddings, audio = batchi
            audio = audio.to(device)
            prediction = model.predict(embeddings)
            if all_prediction is None:
                all_prediction = torch.flatten(prediction)
                all_audio = torch.flatten(audio)
            else:
                all_prediction = torch.cat((all_prediction, torch.flatten(prediction)))
                all_audio = torch.cat((all_audio, torch.flatten(audio)))

        # print(embeddings)
        # print(model(embeddings, audio).shape)
        # print(model.predict(embeddings))
        all_prediction = all_prediction.detach().cpu()
        all_audio = all_audio.detach().cpu()
        
        epoch_acc = accuracy_score(all_audio, all_prediction) 
        epoch_f1 = f1_score(all_audio, all_prediction, average='micro')
        writer.add_scalar('valid_acc', epoch_acc, epoch)
        writer.add_scalar('valid_f1', epoch_f1, epoch)
        print("[Valid]: ACC: {} - F1: {}".format(str(epoch_acc),str(epoch_f1)))

    if epoch_acc > best_acc: 
            best_acc, best_epoch = epoch_acc, epoch
            print("------------Best model, saving...------------")
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            torch.save(model, '{}/{}textlen_{}lr_{}bs_{}frame_{}.pth'.format(
                args.model_path, args.max_text_lenth, str(lrlast), args.batch_size, args.frame, args.log_name))

    if epoch > best_epoch:
        metrics_update += 1
    else:
        metrics_update = 0
    if metrics_update>=5:
        break     


all_prediction = None
all_audio = None
with torch.no_grad():
    for i, batchi in enumerate(test_loader):
        embeddings, audio = batchi
        audio = audio.to(device)
        prediction = model.predict(embeddings)
        if all_prediction is None:
            all_prediction = prediction.reshape(1,-1)
            all_audio = audio.reshape(1,-1)
        else:
            all_prediction = torch.cat((all_prediction, prediction), dim=0)
            all_audio = torch.cat((all_audio, audio), dim=0)
torch.save(all_prediction, "./text_{}frame_prediction.pt".format(args.frame))
torch.save(all_audio, "./text_{}frame_audio.pt".format(args.frame))

print("End. Best epoch {:03d}".format(best_epoch))




