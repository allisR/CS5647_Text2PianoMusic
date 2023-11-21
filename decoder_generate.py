import sys
from torch.optim.lr_scheduler import LambdaLR
import os
from tqdm import tqdm
from parse import parse_args
from audio_decoder import Audio_Decoder
from transformers import AdamW
from utils import *
from sklearn.metrics import accuracy_score  
from sklearn.metrics import f1_score
from dataset import ContextDataset
from transformers import BertGenerationConfig, BertGenerationEncoder, EncoderDecoderModel, BertGenerationDecoder
from third_party.constants import *
from torch.utils.data import Dataset, DataLoader
from dataset import create_epiano_datasets, compute_epiano_accuracy
from transformers import Trainer, TrainingArguments,BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from lr_scheduling import LrStepTracker
if __name__ == '__main__':
    args = parse_args()
    base_path = args.base_path # './maestro-v2.0.0/'
    csv_path = args.data_path + '/clean+GPTcaption.csv' # './data/clean+GPTcaption.csv'
    processed_path = args.data_path + '/max_len{}_data.npy'.format(args.max_len)
    print('generating dataset...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    train_dataset, val_dataset, test_dataset = create_epiano_datasets("data", 2048)
    print('Done...')
    pth_path = '{}/decoder.pth'.format(args.model_path)

    model = Audio_Decoder(target_vocab_size = VOCAB_SIZE, embed_dim = 512, nhead = 8, num_layers = 6, device = device)
    # model = torch.load(pth_path).to(device)
    # torch.save(model.state_dict(), "best_acc_weights.pickle")
    model.load_state_dict(torch.load("best_acc_weights.pickle"))
    model.to(device)
    print("success loaded")
    num_prime = 256
    f = str(random.randrange(len(test_dataset)))
    idx = int(f)
    primer, _  = test_dataset[idx]
    primer = primer.to(device)
    f_path = os.path.join('output_midi', "primer.mid")
    decode_midi(primer[:num_prime].cpu().numpy(), file_path=f_path)
    model.eval()
    with torch.set_grad_enabled(False):
        print("RAND DIST")
        rand_seq = model.predict_decoder(primer[:num_prime], 1024)
        f_path = os.path.join('output_midi', "rand.mid")
        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)

