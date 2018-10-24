import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from voc import Voc
import argparse

def indexesFromSentence(voc, sentence):
    return [voc.SOS_token] + [voc.word2index[word] for word in sentence.split(" ")] + [voc.EOS_token]

def zeroPadding(l, fillvalue):
    # fillvalue = PAD_token
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)

def inputVar():
    pass


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description="input and annFile Path to training mscoco")
    parser.add_argument("--input_path", "-i", type=str, required=True)
    parser.add_argument("--ann_path", "-ann", type=str, required=True)
    args = parser.parse_args()

    # hyper parameters
    hidden_size = 512
    batch_size = 32
    num_words = 50 # TODO : change

    # define DataLoader
    cap = dset.CocoCaptions(root = args.input_path,
                            annFile = args.ann_path,
                            transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()
                                ]))
    dataLoader = DataLoader(cap, batch_size=4, shuffle=True, num_workers=4)

    # load from Voc
    """
    with open("voc.pkl", "rb") as f:
        voc = pickle.load(f)
    """

    # define hyper parameters and models
    hidden_size = 512
    batch_size = 32
    num_words = 50 # 仮

    # using pretrained Model
    # TODO : define LSTM Decoder
    embedding = nn.Embedding(num_words, hidden_size)
    encoder = models.resnet50(pretrained=True)
    num_fits = encoder.fc.in_features
    encoder.fc = nn.Linear(num_fits, hidden_size)

    for i_batch, sampled_batched in enumerate(dataLoader):
        import ipdb; ipdb.set_trace()
