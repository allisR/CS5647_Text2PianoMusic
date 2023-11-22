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
from dataset import ContextDataset, compute_all_metrics
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
    model.load_state_dict(torch.load("all_best_acc.pickle", map_location=device))
    print("successful load model")
    model.to(device)

    def test_predict(test_loader, device, model, write_path = './output_midi/'):
        with torch.no_grad():
            model.eval()
            sum_acc    = 0.0
            sum_precision = 0.0
            sum_recall = 0.0
            sum_f = 0.0
            n_test     = len(test_loader)
            for i, batchi in enumerate(test_loader):
                input_ids, attention_mask, token_type_ids, audio, label = batchi
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                audio = audio.type(torch.LongTensor).to(device)
                label = label.type(torch.LongTensor).to(device)
                # loss = model(input_ids=input_ids, decoder_input_ids = audio, labels=label, attention_mask = attention_mask, return_dict=True).loss
                predictions = model(input_ids, attention_mask, audio, tgt_key_padding_mask=None)
                acc,  precision, recall, f = compute_all_metrics(predictions, label)
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



        # with torch.no_grad():
        #     for i, batchi in enumerate(test_loader):
        #         input_ids, attention_mask, token_type_ids, audio, label = batchi
        #         input_ids = input_ids.to(device)
        #         attention_mask = attention_mask.to(device)
        #         token_type_ids = token_type_ids.to(device)
        #         audio = audio.type(torch.LongTensor).to(device)
        #         label = label.type(torch.LongTensor).to(device)
        #         prediction = model.predict(input_ids, attention_mask, device = device, target_seq_length = 1024)
        #         f_path = os.path.join('output_midi', "text.mid")
        #         decode_midi(prediction[0], file_path=f_path)
        #         break
    
    test_predict(train_loader, device, model)