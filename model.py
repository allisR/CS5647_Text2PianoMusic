import torch
import torch.nn as nn
import torch.optim as optim
from text_encoder import BERT_Encoder
from audio_decoder import Audio_Decoder

# Define the Transformer model
class Audio_Generator(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, decoder_nhead, decoder_num_layers):
        super(Audio_Generator, self).__init__()
        self.text_encoder = BERT_Encoder(hidden=embed_dim)
        self.audio_generator = Audio_Decoder(target_vocab_size = target_vocab_size, embed_dim = embed_dim, nhead = decoder_nhead, num_layers = decoder_num_layers)

    def forward(self, text, audio):
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        text_embed_seq = self.text_encoder(input_ids, attention_mask=attention_mask) # [B, S, E]
        print(text_embed_seq.shape)
        text_embed_seq = torch.permute(text_embed_seq, (1,0,2)) # [S, B, E]
        label = self.audio_generator(text_embed_seq, audio) # [B, S, E]
        return label