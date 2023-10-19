import torch
import torch.nn as nn
import torch.optim as optim

# Define the Transformer model
class Audio_Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, nhead, num_layers):
        super(Audio_Decoder, self).__init__()
        self.embedding_decoder = nn.Embedding(target_vocab_size, embed_dim)
        # self.decoder = nn.LSTM(embed_dim, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(embed_dim, target_vocab_size)

    def forward(self, text_embed, audio):
        audio_embed = self.embedding_decoder(audio)
        audio_embed = torch.permute(audio_embed, (1, 0, 2))
        print(audio_embed.shape, text_embed.shape)
        decoder_h_seq = self.decoder(audio_embed, text_embed) # seq batch embed [S, B, E]
        decoder_h_seq = torch.permute(decoder_h_seq, (1,0,2))
        output = self.out(decoder_h_seq)
        return output