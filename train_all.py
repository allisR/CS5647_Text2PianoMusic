import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import time
import sys
import os
from tqdm import tqdm
from parse import parse_args
from torch.optim import lr_scheduler
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
from sklearn.metrics import accuracy_score  
from sklearn.metrics import f1_score
from dataset import ContextDataset, compute_epiano_accuracy
from transformers import BertGenerationConfig, BertGenerationEncoder, EncoderDecoderModel, BertGenerationDecoder
from third_party.constants import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args = parse_args()
    base_path = args.base_path # './maestro-v2.0.0/'
    csv_path = args.data_path + '/clean+GPTcaption.csv' # './data/clean+GPTcaption.csv'
    processed_path = args.data_path + '/max_len{}_data.npy'.format(args.max_len)

    text_descriptions, audios, padding_masks = load_data_sam(processed_path, csv_path, base_path, max_seq = args.max_len)
    train_text, valid_text, test_text, train_audios, valid_audios, test_audios, train_labels, valid_labels, test_labels = split_train_val_test(text_descriptions, audios, padding_masks)

    print('generating dataset...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = ContextDataset(train_text, tokenizer, args.max_text_lenth, train_audios, train_labels, device)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)

    valid_dataset = ContextDataset(valid_text, tokenizer, args.max_text_lenth, valid_audios, valid_labels, device)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size= args.batch_size, shuffle=False)

    test_dataset = ContextDataset(test_text, tokenizer, args.max_text_lenth, test_audios, test_labels, device)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= args.batch_size, shuffle=False)


    model = Audio_Generator(target_vocab_size = VOCAB_SIZE, embed_dim = 512, decoder_nhead = 8, decoder_num_layers =  6, device = device)
    model.to(device)
    lrlast = 5e-4
    lrmain = 5e-4
    pth_path = '{}/{}textlen_{},{}lr_{}bs_{}frame_{}.pth'.format(
                    args.model_path, args.max_text_lenth, str(lrlast), str(lrmain), args.batch_size, args.frame, args.log_name)
    print(pth_path)
    optimizer = torch.optim.Adam(
        [
            {"params":model.encoder.parameters(),"lr": lrmain},
            {"params":model.decoder.parameters(), "lr": lrlast},
    ])
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    print('starting')
    best_epoch = 0
    best_loss = float('inf')
    best_acc = 0
    metrics_update = 0
    for epoch in range(args.num_epochs):
        model.train()
        start_time = time.time()
        t_loss = 0
        train_num = 0 
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batchi in pbar:
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids, audio, label = batchi
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            audio = audio.type(torch.LongTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            # loss = model(input_ids=input_ids, decoder_input_ids = audio, labels=label, attention_mask = attention_mask, return_dict=True).loss
            predictions = model(input_ids, attention_mask, audio, tgt_key_padding_mask=None)
            label = label.flatten()
            predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1], -1)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            t_loss += loss.detach().item()
            train_num += 1

        print('epoch:', epoch)
        print('train_loss {:.4f}'.format(t_loss/train_num) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        scheduler.step()
        with torch.no_grad():
            model.eval()
            sum_acc    = 0.0
            n_test     = len(valid_loader)
            for i, batchi in enumerate(valid_loader):
                optimizer.zero_grad()
                input_ids, attention_mask, token_type_ids, audio, label = batchi
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                audio = audio.type(torch.LongTensor).to(device)
                label = label.type(torch.LongTensor).to(device)
                # loss = model(input_ids=input_ids, decoder_input_ids = audio, labels=label, attention_mask = attention_mask, return_dict=True).loss
                predictions = model(input_ids, attention_mask, audio, tgt_key_padding_mask=None)
                sum_acc += float(compute_epiano_accuracy(predictions, label))

            epoch_acc = sum_acc / n_test
            # acc_list.append(epoch_acc)
            print("[Valid]: ACC: {} - F1:".format(str(epoch_acc)))

        if epoch_acc > best_acc: 
                best_acc, best_epoch = epoch_acc, epoch
                print("------------Best model, saving...------------")
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model.state_dict(), "all_best_acc.pickle")
                print("------------Finished Save---------------------")

        if epoch > best_epoch:
            metrics_update += 1
        else:
            metrics_update = 0
        if metrics_update>=10:
            break     

    




