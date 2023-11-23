# CS5647_Text2PianoMusic
## Train audio decoder
1. Generate dataset. Run `preprocess_midi.py` in the `utils` folder. This will write pre-processed data into the `data` folder split into `train`, `val`, and `test` as per Maestro's recommendation.
2. Run `train_decoder.py` to train the audio decoder. We provided the pre-trained weight file as `all_best_acc.pickle`.
3. Test the trained audio decoder by running `decoder_generate.py`. The output midis should be in the `output_midi` folder as `primer.mid` and `primer_after.mid`.
## Train whole model
1. Run `train_decoder.py` to train the whole model. We provided the model weight file as `all_best_acc.pickle`.
2. Run `model_generate.py` to test the whole model and generate a midi file based on the text input. The output midi should be in the `output_midi` folder as `text.mid`

