import torch
from torch.utils.data import Dataset, DataLoader

# class ContextDataset(Dataset):
#     def __init__(self, text_inputs, audio_inputs, txt_tokenizer, audio_tokenizer, max_len):
#         self.txt = text_inputs
#         self.audio = audio_inputs
#         self.txt_tokenizer = txt_tokenizer
#         self.max_len = max_len
#         self.audio_tokenizer = audio_tokenizer

#     def __len__(self):
#         return len(self.txt)

#     def audio_tokenizer(self, audio):
#         """

#         """
#         return self.audio_tokenizer(audio)

#     def __getitem__(self, index):
#         audio = self.audio_tokenizer(self.audio[index])
#         text = self.txt[index]

#         embedding = self.txt_tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             return_token_type_ids=True,
#             padding='max_length',
#             return_attention_mask=True,
#             return_tensors='pt',
#             truncation=True
#         )

#         return {
#             "text": text,
#             "input_ids": embedding['input_ids'].flatten(),
#             "attention_mask": embedding['attention_mask'].flatten(),
#             "token_type_ids": embedding['token_type_ids'].flatten(),
#             "labels": audio
#         }


class ContextDataset(Dataset):
    def __init__(self, x_inputs, tokenizer, max_len, audios, padding_masks, device):
        self.x = x_inputs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.audios = audios
        self.padding_masks = padding_masks
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        text = self.x[index]
        audio = self.audios[index]
        padding_mask = self.padding_masks[index]

        embedding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return embedding['input_ids'].flatten(),embedding['attention_mask'].flatten(),embedding['token_type_ids'].flatten(), audio, padding_mask
        # return {"input_ids": embedding['input_ids'].flatten().to(self.device),
        # "attention_mask": embedding['attention_mask'].flatten().to(self.device),
        # "token_type_ids": embedding['token_type_ids'].flatten().to(self.device)
        # }, audio, padding_mask