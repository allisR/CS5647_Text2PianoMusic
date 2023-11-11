import torch
import torch.nn as nn
import torch.optim as optim
import math
# Define the Audio_Decoder model
class Audio_Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, nhead, num_layers):
        super(Audio_Decoder, self).__init__()
        self.d_model = embed_dim
        self.embedding_encoder = nn.Embedding(target_vocab_size, embed_dim)
        # self.decoder = nn.LSTM(embed_dim, embed_dim)
        self.pos_decoder = PositionalEncoding(d_model=embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(embed_dim, target_vocab_size)
        self.src_mask = None
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding_encoder.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, text_embed, audio, memory_key_padding_mask, tgt_key_padding_mask):
        audio = torch.permute(audio, (1,0))
        audio_embed = self.embedding_encoder(audio)* math.sqrt(self.d_model)
        S, B, E =  audio_embed.shape
        audio_embed = self.pos_decoder(audio_embed)
        
        if self.src_mask is None or self.src_mask.size(0) != S:
            device = audio.device
            self.src_mask = nn.Transformer.generate_square_subsequent_mask(S).to(device)

        decoder_h_seq = self.decoder(tgt = audio_embed, memory = text_embed, tgt_mask = self.src_mask, 
                                     tgt_key_padding_mask = tgt_key_padding_mask.type(torch.int64),
                                     memory_key_padding_mask = memory_key_padding_mask) # seq batch embed [S, B, E]
        # decoder_h_seq = self.decoder(audio_embed, self.src_mask) # seq batch embed [S, B, E]
        decoder_h_seq = torch.permute(decoder_h_seq, (1,0,2))
        output = self.out(decoder_h_seq)
        return output

    def predict(self, text_embed, memory_key_padding_mask):
        start = torch.tensor([[129]]).type(torch.LongTensor)
        while True:
            audio_embed = self.embedding_encoder(start)
            audio_embed = self.pos_decoder(audio_embed)
            audio_embed = torch.permute(audio_embed, (1, 0, 2))

            decoder_h_seq = self.decoder(audio_embed, text_embed, memory_key_padding_mask=memory_key_padding_mask)
            decoder_h_seq = torch.permute(decoder_h_seq, (1, 0, 2))
            output = self.out(decoder_h_seq)
            output = torch.argmax(output, dim=2)[0][-1].reshape(1,-1)
            start = torch.cat([start, output], 1)
            if start[0][-1].item() == 130:
                break
        return start


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