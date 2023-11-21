import torch
import torch.nn as nn
import torch.optim as optim
import math
from third_party.constants import *
import random
# Define the Audio_Decoder model
class Audio_Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, nhead, num_layers, device):
        super(Audio_Decoder, self).__init__()
        self.d_model = embed_dim
        self.embedding_encoder = nn.Embedding(target_vocab_size, embed_dim)
        # self.decoder = nn.LSTM(embed_dim, embed_dim)
        self.pos_decoder = PositionalEncoding(d_model=embed_dim, max_len = 2048)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward = 1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(embed_dim, target_vocab_size)
        self.src_mask = None
        # self.init_weights()
        self.device = device
        self.softmax = nn.Softmax(dim=-1)

    # def init_weights(self):
    #     initrange = 1.0
    #     self.embedding_encoder.weight.data.uniform_(-initrange, initrange)
    #     self.out.bias.data.zero_()
    #     self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, text_embed, audio, memory_key_padding_mask, tgt_key_padding_mask):
        audio_embed = self.embedding_encoder(audio)
        audio_embed =  audio_embed.permute(1,0,2)
        audio_embed = self.pos_decoder(audio_embed)
        device = audio.device
        S, B, E = audio_embed.shape
        if self.src_mask is None or self.src_mask.size(0) != S:
            self.src_mask = nn.Transformer.generate_square_subsequent_mask(S).to(device)

        if tgt_key_padding_mask is None:
            # memory_key_padding_mask = ~memory_key_padding_mask.type(torch.bool)
            if text_embed is None:
                decoder_h_seq = self.decoder(tgt = audio_embed, memory = torch.ones(1, B, E).to(self.device), tgt_mask = self.src_mask)
            else:
                decoder_h_seq = self.decoder(tgt = audio_embed, memory = text_embed, tgt_mask = self.src_mask)
        else:
            decoder_h_seq = self.decoder(tgt = audio_embed, memory = text_embed, tgt_mask = self.src_mask, 
                                     tgt_key_padding_mask = tgt_key_padding_mask.type(torch.bool),
                                     memory_key_padding_mask = ~memory_key_padding_mask.type(torch.bool)) # seq batch embed [S, B, E]
        # decoder_h_seq = self.decoder(audio_embed, self.src_mask) # seq batch embed [S, B, E]
        decoder_h_seq = torch.permute(decoder_h_seq, (1,0,2))
        output = self.out(decoder_h_seq)
        return output

    def predict_decoder(self, primer, target_seq_length=1024):
        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=torch.device('cuda:0'))
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(torch.device('cuda:0'))
        cur_i = num_primer
        while(cur_i < target_seq_length):
            y = self.softmax(self.forward(text_embed = None, memory_key_padding_mask = None, tgt_key_padding_mask = None, audio = gen_seq[..., :cur_i]))[..., :TOKEN_END]
            token_probs = y[:, cur_i-1, :]
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            # print("next token:",next_token)
            gen_seq[:, cur_i] = next_token


            # Let the transformer decide to end if it wants to
            if(next_token == TOKEN_END):
                print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                break
            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        return gen_seq[:, :cur_i]


    def predict(self, text_embed, memory_key_padding_mask, device, target_seq_length=1024):
        ix = random.randrange(128)
        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=torch.device('cuda:0'))
        gen_seq[:, :1] = ix
        cur_i = 1
        while(cur_i < target_seq_length):
            y = self.softmax(self.forward(text_embed = text_embed, memory_key_padding_mask = memory_key_padding_mask, tgt_key_padding_mask = None, audio = gen_seq[..., :cur_i]))[..., :TOKEN_END]
            token_probs = y[:, cur_i-1, :]
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            # print("next token:",next_token)
            gen_seq[:, cur_i] = next_token

            # Let the transformer decide to end if it wants to
            if(next_token == TOKEN_END):
                print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                break
            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        return gen_seq[:, :cur_i]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



