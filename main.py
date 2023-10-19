import torch
import torch.nn as nn
import torch.optim as optim


decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(30, 32, 512)
out = transformer_decoder(tgt, memory) # seq batch embed
print(out.shape)