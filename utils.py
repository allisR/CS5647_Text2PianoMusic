# import os
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
        frame = int(frame)
        audios = []
        max_end = 0

        for f in midi_files:
            pm = PrettyMIDI(f)
            instrument = pm.instruments[0]
            notes, end = GetNoteSequence(instrument)
            max_end = max(max_end, end)
            audios.append(notes)
        output_audios = np.zeros([len(audios),int(max_end)*frame+frame])
        for i,j in enumerate(audios):
             for k in j:
                output_audios[i][int(k[1]*frame):int(k[2]*frame)] = k[0]
        return output_audios

