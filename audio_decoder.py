import torch
import torch.nn as nn
import torch.optim as optim

# Define the Transformer model
class Audio_Generator(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers):
        super(Audio_Generator, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src, mask):
        output = self.transformer(src, mask)
        output = self.fc(output)
        return output