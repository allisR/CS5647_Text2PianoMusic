import torch
import torch.nn as nn
import torch.optim as optim
from text_encoder import BERT_Encoder
from audio_decoder import Audio_Decoder

# Define the Transformer model
class Audio_Generator(nn.Module):
    def __init__(self, input_dim, output_dim, target_vocab_size, embed_dim, decoder_nhead, decoder_num_layers):
        super(Audio_Generator, self).__init__()
        self.text_encoder = BERT_Encoder(hidden=1000)
        self.audio_generator = Audio_Decoder(target_vocab_size = target_vocab_size, embed_dim = embed_dim, nhead = decoder_nhead, num_layers = decoder_num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, text, audio):
        text_embed_seq = self.text_encoder(text) # [B, S, E]
        text_embed_seq = torch.permute(text_embed_seq, (1,0,2)) # [S, B, E]
        label = self.audio_generator(text_embed_seq, audio) # [B, S, E]
        return label