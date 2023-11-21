import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("--base_path",
        type=str,
        default="./maestro-v2.0.0/",
        help="base_path of audio data")
    parser.add_argument("--data_path",
        type=str,
        default="./data",
        help="dataset path")
    parser.add_argument("--model_path",
        type=str,
        default="./models",
        help="model save path")
    parser.add_argument("--log_name",
        type=str,
        default="log",
        help="log name")
    parser.add_argument("--max_text_lenth",
        type=int,
        default=128,
        help="max_lenth for bert tokenizer")
    parser.add_argument("--batch_size",
        type=int,
        default=4,
        help="batch size for dataloader")
    parser.add_argument("--frame",
        type=int,
        default=5,
        help="#frame/sec for audio")
    parser.add_argument("--num_epochs",
        type=int,
        default=20,
        help="#of epochs for training")
    parser.add_argument("--max_len",
        type=int,
        default=2048,
        help="#of epochs for training")
    


    return parser.parse_args()