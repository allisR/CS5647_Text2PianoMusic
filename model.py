import torch
import torch.nn as nn
import torch.optim as optim
from text_encoder import BERT_Encoder
from audio_decoder import Audio_Decoder

# Define the Transformer model
class Audio_Generator(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, decoder_nhead, decoder_num_layers, device):
        super(Audio_Generator, self).__init__()
        self.encoder = BERT_Encoder(hidden=embed_dim)
        self.decoder = Audio_Decoder(target_vocab_size = target_vocab_size, embed_dim = embed_dim, nhead = decoder_nhead, num_layers = decoder_num_layers, device = device)
        self.decoder.load_state_dict(torch.load("best_acc_weights.pickle", map_location=device))
        print("Successful loaded decoder!")

    def forward(self, input_ids, attention_mask, audio, tgt_key_padding_mask):
        text_embed_seq = self.encoder(input_ids, attention_mask=attention_mask)
        # text_embed_seq = self.encoder(input_ids, attention_mask=attention_mask).unsqueeze(1) # [B, S, E]
        text_embed_seq = torch.permute(text_embed_seq, (1,0,2)) # [S, B, E]
        label = self.decoder(text_embed_seq, audio, memory_key_padding_mask = attention_mask, tgt_key_padding_mask = tgt_key_padding_mask) # [B, S, E]
        return label

    def predict(self, input_ids, attention_mask, device, target_seq_length):
        all_result = []
        for i in range(len(input_ids)):
            one_id = input_ids[i, :].unsqueeze(0)
            one_att_mask = attention_mask[i, :].unsqueeze(0)
            text_embed_seq = self.encoder(one_id, attention_mask=one_att_mask)  # [B, S, E]
            text_embed_seq = torch.permute(text_embed_seq, (1, 0, 2))  # [S, B, E]
            result = self.decoder.predict(text_embed_seq, memory_key_padding_mask = one_att_mask, device = device, target_seq_length = target_seq_length).detach().cpu().numpy()
            all_result.append(result[0])
            break
        return all_result


