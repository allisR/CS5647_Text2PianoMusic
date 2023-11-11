import os
# import sys
# import json
# import soundfile as sf
import pandas as pd
import numpy as np
from pretty_midi import PrettyMIDI


def read_csv(path):
    data = pd.read_csv(path)
    return data


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
            padding_masks[i][:int(k[2]*frame)+1] = 1
        return output_audios, padding_masks

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