import torch
import os
import threading
import time
import tkinter.messagebox
from parse import parse_args
from transformers import BertTokenizer
from model import Audio_Generator
from utils import *
from dataset import ContextDataset
from third_party.constants import *
from ttkthemes import themed_tk as tk
from midi2audio import FluidSynth
from pygame import mixer
from tkinter import *
from tkinter import ttk


args = parse_args()
write_path = './output_midi/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = Audio_Generator(target_vocab_size = VOCAB_SIZE, embed_dim = 512, decoder_nhead = 8, decoder_num_layers =  6, device = device)
model.load_state_dict(torch.load("all_best_acc.pickle"))
print("successful load model")
model.to(device)

def test_predict(text, args, tokenizer, device, model, write_path = './output_midi/'):
    with torch.no_grad():
        embedding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=args.max_text_lenth,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids, attention_mask, token_type_ids = embedding['input_ids'].flatten(),embedding['attention_mask'].flatten(),embedding['token_type_ids'].flatten()
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        token_type_ids = token_type_ids.unsqueeze(0).to(device)
        prediction = model.predict(input_ids, attention_mask, device = device, target_seq_length = 1024)
        f_path = os.path.join('output_midi', "text.mid")
        decode_midi(prediction[0], file_path=f_path)

root = tk.ThemedTk()
root.get_themes()                 # Returns a list of all themes that can be set
root.set_theme("radiance")         # Sets an available theme

# Fonts - Arial (corresponds to Helvetica), Courier New (Courier), Comic Sans MS, Fixedsys,
# MS Sans Serif, MS Serif, Symbol, System, Times New Roman (Times), and Verdana
#
# Styles - normal, bold, roman, italic, underline, and overstrike.

statusbar = ttk.Label(root, text="Welcome to Melody", relief=SUNKEN, anchor=W, font='Times 10 italic')
statusbar.pack(side=BOTTOM, fill=X)

# Create the menubar
menubar = Menu(root)
root.config(menu=menubar)
root.geometry('1100x400')
# Create the submenu

subMenu = Menu(menubar, tearoff=0)

playsong = write_path + 'output.wav'


menubar.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="Exit", command=root.destroy)

subMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help", menu=subMenu)

mixer.init()  # initializing the mixer

root.title("Melody")
root.iconbitmap(r'images/melody.ico')

# Root Window - StatusBar, LeftFrame, RightFrame
# LeftFrame - The listbox (playlist)
# RightFrame - TopFrame,MiddleFrame and the BottomFrame


leftframe = Frame(root)
leftframe.pack(side=LEFT, padx=30, pady=30)

generatelabel = Label(leftframe, text= "Please write some description to generate music!")
generatelabel.pack()

textExample=tkinter.Text(leftframe, height=10)     #创建文本输入框
textExample.pack()   #把Text放在window上面，显示Text这个控件

def generate_from_text():
    global generatelabel, textExample, Generatebtn, playBtn, pauseBtn, rewindBtn, volumeBtn
    text=textExample.get("1.0","end")    #获取文本输入框的内容
    Generatebtn["state"] = DISABLED
    playBtn["state"] = DISABLED
    pauseBtn["state"] = DISABLED
    rewindBtn["state"] = DISABLED
    volumeBtn["state"] = DISABLED
    test_predict(text, args, tokenizer, device, model, write_path)
    fs = FluidSynth(sound_font='GeneralUser.sf2')
    # fs.play_midi(write_path + 'text.mid')
    fs.midi_to_audio(write_path + 'text.mid', write_path + 'output.wav') 
    generatelabel.configure(text="Generate success! You can listen now!")
    Generatebtn["state"] = NORMAL
    playBtn["state"] = NORMAL
    pauseBtn["state"] = NORMAL
    rewindBtn["state"] = NORMAL
    volumeBtn["state"] = NORMAL
 
Generatebtn=ttk.Button(leftframe, text="Generate", command=generate_from_text)   #command绑定获取文本框内容的方法
Generatebtn.pack()


rightframe = Frame(root)
rightframe.pack(pady=30)

topframe = Frame(rightframe)
topframe.pack()

lengthlabel = ttk.Label(topframe, text='Total Length : --:--')
lengthlabel.pack(pady=5)

currenttimelabel = ttk.Label(topframe, text='Current Time : --:--', relief=GROOVE)
currenttimelabel.pack()


def show_details(play_song):
    a = mixer.Sound(play_song)
    total_length = a.get_length()

    # div - total_length/60, mod - total_length % 60
    mins, secs = divmod(total_length, 60)
    mins = round(mins)
    secs = round(secs)
    timeformat = '{:02d}:{:02d}'.format(mins, secs)
    lengthlabel['text'] = "Total Length" + ' - ' + timeformat

    t1 = threading.Thread(target=start_count, args=(total_length,))
    t1.start()


def start_count(t):
    global paused
    # mixer.music.get_busy(): - Returns FALSE when we press the stop button (music stop playing)
    # Continue - Ignores all of the statements below it. We check if music is paused or not.
    current_time = 0
    while current_time <= t and mixer.music.get_busy():
        if paused:
            continue
        else:
            mins, secs = divmod(current_time, 60)
            mins = round(mins)
            secs = round(secs)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            currenttimelabel['text'] = "Current Time" + ' - ' + timeformat
            time.sleep(1)
            current_time += 1


def play_music():
    global paused

    if paused:
        mixer.music.unpause()
        statusbar['text'] = "Music Resumed"
        paused = FALSE
    else:
        try:
            stop_music()
            time.sleep(0.5)
            play_it = playsong
            mixer.music.load(play_it)
            mixer.music.play()
            statusbar['text'] = "Playing music" + ' - ' + os.path.basename(play_it)
            show_details(play_it)
        except:
            tkinter.messagebox.showerror('File not found', 'Please generate a song first.')


def stop_music():
    mixer.music.stop()
    statusbar['text'] = "Music Stopped"


paused = FALSE


def pause_music():
    global paused
    paused = TRUE
    mixer.music.pause()
    statusbar['text'] = "Music Paused"


def rewind_music():
    play_music()
    statusbar['text'] = "Music Rewinded"


def set_vol(val):
    volume = float(val) / 100
    mixer.music.set_volume(volume)
    # set_volume of mixer takes value only from 0 to 1. Example - 0, 0.1,0.55,0.54.0.99,1


muted = FALSE


def mute_music():
    global muted
    if muted:  # Unmute the music
        mixer.music.set_volume(0.7)
        volumeBtn.configure(image=volumePhoto)
        scale.set(70)
        muted = FALSE
    else:  # mute the music
        mixer.music.set_volume(0)
        volumeBtn.configure(image=mutePhoto)
        scale.set(0)
        muted = TRUE


middleframe = Frame(rightframe)
middleframe.pack(pady=30, padx=30)

playPhoto = PhotoImage(file='images/play.png')
playBtn = ttk.Button(middleframe, image=playPhoto, command=play_music)
playBtn.grid(row=0, column=0, padx=10)

pausePhoto = PhotoImage(file='images/pause.png')
pauseBtn = ttk.Button(middleframe, image=pausePhoto, command=pause_music)
pauseBtn.grid(row=0, column=2, padx=10)

# Bottom Frame for volume, rewind, mute etc.

bottomframe = Frame(rightframe)
bottomframe.pack()

rewindPhoto = PhotoImage(file='images/rewind.png')
rewindBtn = ttk.Button(bottomframe, image=rewindPhoto, command=rewind_music)
rewindBtn.grid(row=0, column=0)

mutePhoto = PhotoImage(file='images/mute.png')
volumePhoto = PhotoImage(file='images/volume.png')
volumeBtn = ttk.Button(bottomframe, image=volumePhoto, command=mute_music)
volumeBtn.grid(row=0, column=1)

scale = ttk.Scale(bottomframe, from_=0, to=100, orient=HORIZONTAL, command=set_vol)
scale.set(70)  # implement the default value of scale when music player starts
mixer.music.set_volume(0.7)
scale.grid(row=0, column=2, pady=15, padx=30)


def on_closing():
    stop_music()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
