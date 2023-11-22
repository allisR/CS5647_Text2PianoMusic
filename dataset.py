import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from third_party.constants import *
from sklearn.metrics import precision_recall_fscore_support

SEQUENCE_START = 0

# EPianoDataset
class EPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        ----------
        """

        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_mid     = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE, device=torch.device('cpu'))
        i_stream.close()

        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

        return x, tgt

# process_midi
def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=torch.device('cpu'))
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=torch.device('cpu'))

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len-1]      = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]

    return x, tgt


# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq, random_seq=True):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------
    """

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq, random_seq)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset


def compute_all_metrics(out, tgt):
    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)
    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0
    
    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)
    precision, recall, f, _ = precision_recall_fscore_support(tgt.detach().cpu().numpy(), out.detach().cpu().numpy(), average='macro')

    return float(acc), precision, recall, f


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


