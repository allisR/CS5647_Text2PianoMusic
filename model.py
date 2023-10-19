import torch
import torch.nn as nn
import torch.optim as optim
from text_encoder import BERT_Encoder
from audio_decoder import Audio_Generator

# Define the Transformer model
class Audio_Generator(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers):
        super(Audio_Generator, self).__init__()
        self.text_encoder = BERT_Encoder(hidden=1000)
        self.audio_generator = Audio_Generator(input_dim=1000, output_dim=output_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, text):
        text_embed = self.text_encoder(text)
        audio = self.audio_generator(text_embed)
        return audio