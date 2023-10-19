import torch
from torch.utils.data import Dataset, DataLoader

class ContextDataset(Dataset):
    def __init__(self, text_inputs, audio_inputs, txt_tokenizer, audio_tokenizer, max_len):
        self.txt = text_inputs
        self.audio = audio_inputs
        self.txt_tokenizer = txt_tokenizer
        self.max_len = max_len
        self.audio_tokenizer = audio_tokenizer

    def __len__(self):
        return len(self.txt)

    def audio_tokenizer(self, audio):
        """

        """
        return self.audio_tokenizer(audio)

    def __getitem__(self, index):
        audio = self.audio_tokenizer(self.audio[index])
        text = self.txt[index]

        embedding = self.txt_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            "text": text,
            "input_ids": embedding['input_ids'].flatten(),
            "attention_mask": embedding['attention_mask'].flatten(),
            "token_type_ids": embedding['token_type_ids'].flatten(),
            "labels": audio
        }
