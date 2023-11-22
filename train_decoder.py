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
from torch.optim.lr_scheduler import LambdaLR
import os
from tqdm import tqdm
from parse import parse_args
from audio_decoder import Audio_Decoder
from transformers import AdamW
from utils import *
from sklearn.metrics import accuracy_score  
from sklearn.metrics import f1_score
from dataset import ContextDataset
from transformers import BertGenerationConfig, BertGenerationEncoder, EncoderDecoderModel, BertGenerationDecoder
from third_party.constants import *
from torch.utils.data import Dataset, DataLoader
from dataset import compute_all_metrics, create_epiano_datasets
from transformers import Trainer, TrainingArguments,BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from lr_scheduling import LrStepTracker
if __name__ == '__main__':
    args = parse_args()
    base_path = args.base_path # './maestro-v2.0.0/'
    csv_path = args.data_path + '/clean+GPTcaption.csv' # './data/clean+GPTcaption.csv'
    processed_path = args.data_path + '/max_len{}_data.npy'.format(args.max_len)
    print('generating dataset...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    train_dataset, val_dataset, test_dataset = create_epiano_datasets("data", 2048)
    print('Done...')
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5)
    test_loader = DataLoader(test_dataset, batch_size=5)


    model = Audio_Decoder(target_vocab_size = VOCAB_SIZE, embed_dim = 512, nhead = 8, num_layers = 6, device = device)
    # model = MusicTransformer(n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
    #              dropout=0.1, max_sequence=2048)
    model.to(device)
    pth_path = '{}/decoder.pth'.format(args.model_path)
    init_step = 0
    lr_stepper = LrStepTracker(512, SCHEDULER_WARMUP_STEPS, init_step)
    optimizer = AdamW(model.parameters(), lr= 1)
    lr_scheduler = LambdaLR(optimizer, lr_stepper.step)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    print('starting')
    best_epoch = 0
    best_loss = float('inf')
    best_acc = 0
    metrics_update = 0
    acc_list = []
    for epoch in range(args.num_epochs):
        model.train()
        start_time = time.time()
        t_loss = 0
        train_num = 0 
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in pbar:
            optimizer.zero_grad()
            x   = batch[0].to(device)
            tgt = batch[1].to(device)
            y = model(text_embed = None, audio = x, memory_key_padding_mask =None ,tgt_key_padding_mask=None)
            # y = model(x)
            y = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()
            loss = criterion(y, tgt)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            t_loss += loss.detach().item()
            train_num += 1

        print('epoch:', epoch)
        print('train_loss {:.4f}'.format(t_loss/train_num) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        with torch.no_grad():
            model.eval()
            sum_acc    = 0.0
            sum_precision = 0.0
            sum_recall = 0.0
            sum_f = 0.0
            n_test     = len(val_loader)
            for i, batchi in enumerate(val_loader):
                x   = batch[0].to(device)
                label = batch[1].to(device)
                y = model(text_embed = None, audio = x, memory_key_padding_mask =None ,tgt_key_padding_mask=None)
                # y = model(x)
                acc,  precision, recall, f = compute_all_metrics(y, label)
                sum_acc += acc
                sum_precision += precision
                sum_recall += recall
                sum_f += f

            epoch_acc = sum_acc / n_test
            epoch_pre = sum_precision / n_test
            epoch_recall = sum_recall / n_test
            epoch_f = sum_f / n_test
            # acc_list.append(epoch_acc)
            print("[Valid]: ACC: {} - Precision: {} Recall: {} F1:{}".format(str(epoch_acc), str(epoch_pre), str(epoch_recall), str(epoch_f)))


        if epoch_acc > best_acc: 
                best_acc, best_epoch = epoch_acc, epoch
                print("------------Best model, saving...------------")
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model, pth_path)
                print("------------Finished Save---------------------")

        if epoch > best_epoch:
            metrics_update += 1
        else:
            metrics_update = 0
        # if metrics_update>=10:
        #     break    
    np.save('acc_log.npy', np.array(acc_list))