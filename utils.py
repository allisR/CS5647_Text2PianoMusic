import os
# import sys
# import json
# import soundfile as sf
import pandas as pd
import numpy as np
from pretty_midi import PrettyMIDI
import pretty_midi
from third_party.processor import encode_midi, decode_midi
from third_party.constants import *
SEQUENCE_START = 0
import random 

def read_csv(path):
    data = pd.read_csv(path)
    return data

def cpu_device():
    return torch.device("cpu")

def get_midi_files(data, base_path):
    midi_files = [base_path+i for i in data['midi_filename'].tolist()]
    return midi_files


def get_text_descriptions(data):
    text_descriptions = data['caption'].tolist()
    return text_descriptions


def GetNoteSequence(instrument) -> np.ndarray:
    sorted_notes = sorted(instrument.notes, key=lambda x: x.start)
    assert len(sorted_notes) > 0
    notes = []

    for note in sorted_notes:
        notes.append([int(note.pitch), note.start, note.end])
    # prev_start = sorted_notes[0].start
    # for note in sorted_notes:
    #     notes.append([note.pitch, note.start -
    #                  prev_start, note.end-note.start])
    #     prev_start = note.start
    return notes, note.end


# transform from [pitch,start,end] to [pitch] for each frame
def midi_files_to_audios(midi_files, frame = 100):
        # 128 as start index, 129 as end index
        start_idx = 129
        end_idx = 130
        pad_idx = 131
        frame = int(frame)
        audios = []
        max_end = 0

        for f in midi_files:
            pm = PrettyMIDI(f)
            instrument = pm.instruments[0]
            notes, end = GetNoteSequence(instrument)
            max_end = max(max_end, end)
            audios.append(notes)
        output_audios = np.zeros([len(audios),int(max_end)*frame+frame+2])
        padding_masks = np.zeros([len(audios),int(max_end)*frame+frame+2])
        for i,j in enumerate(audios):
            for k in j:
                output_audios[i][int(k[1]*frame)+1:int(k[2]*frame)+1] = k[0] + 1
            output_audios[i][int(k[2]*frame)] = end_idx
            output_audios[i][0] = start_idx
            padding_masks[i][int(k[2]*frame)+1:] = 1
            output_audios[i][int(k[2]*frame)+1:] = pad_idx
        return output_audios, padding_masks

def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = np.full((max_seq, ), TOKEN_PAD)
    tgt = np.full((max_seq, ), TOKEN_PAD)

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



def midi_files_to_audios_sam(midi_files, max_seq = 2048, random_seq=True):
        audios = []
        labels = []
        padding_mask = []
        for f in midi_files:
            raw_mid = encode_midi(f)
            x, tgt = process_midi(raw_mid, max_seq, random_seq)
            audios.append(x)
            labels.append(tgt)

        return audios, labels

def load_data_sam(processed_path, csv_path, base_path, max_seq):
    if not os.path.exists(processed_path):
        print('processing data...')
        data = read_csv(csv_path)
        midi_files = get_midi_files(data, base_path)
        text_descriptions = get_text_descriptions(data)
        audios, labels = midi_files_to_audios_sam(midi_files, max_seq)
        data_dict = {'text_descriptions':text_descriptions, 'audios':audios, 'labels':labels}
        np.save(processed_path, data_dict)
    else:
        print('loading data...')
        data_dict = np.load(processed_path, allow_pickle=True).item()
        text_descriptions, audios, labels = data_dict['text_descriptions'], data_dict['audios'], data_dict['labels']
        # print(np.sum(audios[0][:int(np.sum(padding_masks[0]))] == 0), np.sum(padding_masks[0]) )
    return text_descriptions, np.array(audios), np.array(labels)


def load_data(processed_path, csv_path, base_path, frame):
    if not os.path.exists(processed_path):
        print('processing data...')
        data = read_csv(csv_path)
        midi_files = get_midi_files(data, base_path)
        text_descriptions = get_text_descriptions(data)
        audios, padding_masks = midi_files_to_audios(midi_files,frame)
        data_dict = {'text_descriptions':text_descriptions, 'audios':audios, 'padding_masks':padding_masks}
        np.save(processed_path, data_dict)
    else:
        print('loading data...')
        data_dict = np.load(processed_path, allow_pickle=True).item()
        text_descriptions, audios, padding_masks = data_dict['text_descriptions'], data_dict['audios'], data_dict['padding_masks']
        # print(np.sum(audios[0][:int(np.sum(padding_masks[0]))] == 0), np.sum(padding_masks[0]) )
    return text_descriptions, audios, padding_masks


def split_train_val_test(text_descriptions, audios, padding_masks, train_ratio = 0.8):
    num_data = len(text_descriptions)
    train_num = int(num_data * train_ratio)
    valid_num = int((num_data-train_num)/2)

    train_text = text_descriptions[:train_num] # [train_num]
    valid_text = text_descriptions[train_num:train_num+valid_num]
    test_text = text_descriptions[train_num+valid_num:]

    train_audios = audios[:train_num] # [train_num, frame*time]
    valid_audios = audios[train_num:train_num+valid_num]
    test_audios = audios[train_num+valid_num:]

    train_padding_masks = padding_masks[:train_num] # [train_num, frame*time]
    valid_padding_masks = padding_masks[train_num:train_num+valid_num]
    test_padding_masks = padding_masks[train_num+valid_num:]

    return train_text, valid_text, test_text, train_audios, valid_audios, test_audios, train_padding_masks, valid_padding_masks, test_padding_masks



def audios_to_midi_files(frames, index, frame = 100, write_path = './output_midi/'):
        # 128 as start index, 129 as end index
        start_idx = 129
        end_idx = 130
        pad_idx = 131
        frame = int(frame)
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        count = 0
        for mi in frames:
            music = PrettyMIDI()
            piano_program = pretty_midi.instrument_name_to_program(
            'Acoustic Grand Piano')
            piano = pretty_midi.Instrument(program=piano_program)
            t = 0
            mi = mi.cpu().numpy()
            for pitch in mi:
                pitch = int(pitch)
                if pitch < 129 and pitch > 0:
                    note = pretty_midi.Note(
                        velocity=100, pitch=pitch-1, start=t, end=t + 1/frame)
                    t += 1/frame
                    piano.notes.append(note)
                elif pitch == end_idx:
                    break
                else:
                    t += 1/frame

            music.instruments.append(piano)
            music.write(write_path + 'test{}.mid'.format(index+count))
            count += 1