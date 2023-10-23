import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("--base_path",
        type=str,
        default="./maestro-v2.0.0/",
        help="base_path of audio data")
    parser.add_argument("--csv_path",
        type=str,
        default="./data/clean+GPTcaption.csv",
        help="csv dataset path")
    parser.add_argument("--max_text_lenth",
        type=int,
        default=512,
        help="max_lenth for bert tokenizer")
    parser.add_argument("--batch_size",
        type=int,
        default=5,
        help="batch size for dataloader")
    parser.add_argument("--frame",
        type=int,
        default=100,
        help="#frame/sec for audio")

    return parser.parse_args()