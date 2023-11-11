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
from dataset import ContextDataset


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    args = parse_args()
    base_path = args.base_path # './maestro-v2.0.0/'
    csv_path = args.data_path + '/clean+GPTcaption.csv' # './data/clean+GPTcaption.csv'
    processed_path = args.data_path + '/frame{}_data.npy'.format(args.frame)

    writer = SummaryWriter('./tensorboard/log')

    text_descriptions, audios, padding_masks = load_data(processed_path, csv_path, base_path, args.frame)
    train_text, valid_text, test_text, train_audios, valid_audios, test_audios, train_padding_masks, valid_padding_masks, test_padding_masks = split_train_val_test(text_descriptions, audios, padding_masks)


    print('generating dataset...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = ContextDataset(train_text, tokenizer, args.max_text_lenth, train_audios, train_padding_masks, device)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)

    valid_dataset = ContextDataset(valid_text, tokenizer, args.max_text_lenth, valid_audios, valid_padding_masks, device)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size= args.batch_size, shuffle=False)

    test_dataset = ContextDataset(test_text, tokenizer, args.max_text_lenth, test_audios, test_padding_masks, device)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= args.batch_size, shuffle=False)

    # audio_example = torch.randint(0, 10, (1, 8)) # B S

    model = Audio_Generator(target_vocab_size = 128+3, embed_dim = 100, decoder_nhead = 2, decoder_num_layers = 1)

    model.to(device)
    lrlast = .0001
    lrmain = .00001
    pth_path = '{}/{}textlen_{}lr_{}bs_{}frame_{}.pth'.format(
                    args.model_path, args.max_text_lenth, str(lrlast), args.batch_size, args.frame, args.log_name)
    
    optimizer = torch.optim.Adam(
        [
            {"params":model.text_encoder.bert.parameters(),"lr": lrmain},
            {"params":model.text_encoder.mlp.parameters(),"lr": lrlast},
            {"params":model.audio_generator.parameters(), "lr": lrlast},

    ])

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
        t_loss = 0
        train_num = 0 
        for i, batchi in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, audio, padding_mask = batchi
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            audio = audio.type(torch.LongTensor).to(device)
            padding_mask = padding_mask.type(torch.bool).to(device)
            prediction = model(input_ids, attention_mask, audio, padding_mask)

            bs, seq_length, vocab_size = prediction.shape
            scores =   prediction.view(bs*seq_length , vocab_size)  
            audio_label =  audio.view(bs*seq_length )   

            loss = criterion(scores, audio_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.detach().item()
            train_num += 1


        print('epoch:', epoch)
        writer.add_scalar('train_loss', t_loss/train_num, epoch)
        print('train_loss {:.4f}'.format(t_loss/train_num) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        scheduler.step()
        with torch.no_grad():
            all_prediction = None
            all_audio = None
            for i, batchi in enumerate(valid_loader):
                input_ids, attention_mask, token_type_ids, audio, padding_mask = batchi
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                audio = audio.type(torch.LongTensor).to(device)
                padding_mask = padding_mask.type(torch.bool).to(device)
                prediction = model(input_ids, attention_mask, audio, padding_mask)

                prediction = torch.argmax(prediction, dim=2)
                prediction = prediction[padding_mask]
                audio = audio[padding_mask]
                if all_prediction is None:
                    all_prediction = prediction.reshape(-1, )
                    all_audio = audio.reshape(-1, )
                else:
                    prediction = prediction.reshape(-1,)
                    audio = audio.reshape(-1, )
                    all_prediction = torch.cat((all_prediction, prediction))
                    all_audio = torch.cat((all_audio, audio))

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
                torch.save(model, pth_path)
                print("------------Finished Save---------------------")

        if epoch > best_epoch:
            metrics_update += 1
        else:
            metrics_update = 0
        if metrics_update>=5:
            break     

    
    if os.path.exists(pth_path):
        print("------------Load best model...------------")
        model = torch.load(pth_path)
        print("------------Finished------------")
    
    all_prediction = None
    all_audio = None
    with torch.no_grad():
        for i, batchi in enumerate(test_loader):
            input_ids, attention_mask, token_type_ids, audio, padding_mask = batchi
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            audio = audio.type(torch.LongTensor).to(device)
            padding_mask = padding_mask.type(torch.bool).to(device)
            prediction = model(input_ids, attention_mask, audio, padding_mask)

            prediction = torch.argmax(prediction, dim=2)
            prediction = prediction[padding_mask]
            audio = audio[padding_mask]
            if all_prediction is None:
                all_prediction = prediction.reshape(-1, )
                all_audio = audio.reshape(-1, )
            else:
                prediction = prediction.reshape(-1,)
                audio = audio.reshape(-1, )
                all_prediction = torch.cat((all_prediction, prediction), dim=0)
                all_audio = torch.cat((all_audio, audio), dim=0)

    all_prediction = all_prediction.detach().cpu()
    all_audio = all_audio.detach().cpu()
    epoch_acc = accuracy_score(all_audio, all_prediction) 
    epoch_f1 = f1_score(all_audio, all_prediction, average='micro')
    print("[Test]: ACC: {} - F1: {}".format(str(epoch_acc),str(epoch_f1)))   
    # torch.save(all_prediction, "./text_{}frame_prediction.pt".format(args.frame))
    # torch.save(all_audio, "./text_{}frame_audio.pt".format(args.frame))
    # print("End. Best epoch {:03d}".format(best_epoch))




