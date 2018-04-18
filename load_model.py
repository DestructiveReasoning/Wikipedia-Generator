import sys
import json
from os import path

import torch

from main import DATADIR, MODELDIR, VOCABFNAME, HIDDEN_SIZE
from encoder_decoder import EncoderRNN, DecoderRNN
from train import getArticleAbstractPairs
from vocab import loadvocab


VALSET = path.join(DATADIR, 'val.bin')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: puth load_model.py experiment.json [epoch_num]")
        sys.exit(-1)

    # Check if there's a specified epoch, otherwise use the last one by default
    if len(sys.argv) == 3:
        epoch = int(sys.argv[2])
    else:
        epoch = 4

    EXPERIMENT = sys.argv[1]
    if not path.isfile(EXPERIMENT):
        print("%s does not exist" % EXPERIMENT)
        sys.exit(-1)

    with open(EXPERIMENT, 'r') as f:
        experiment = json.load(f)

    print("Loading Vocab...")
    with open(VOCABFNAME, 'r') as f:
        vocab = loadvocab(f, experiment['vocab_size'])

    print("Loading Validation Set...")
    with open(VALSET, 'r') as f:
        pairs = getArticleAbstractPairs(f, vocab)

    encoder = EncoderRNN(vocab.size(), HIDDEN_SIZE)
    decoder = DecoderRNN(2*HIDDEN_SIZE, vocab.size())

    exp_fname = path.split(EXPERIMENT)[-1].split('.')[0]
    model_path = path.join(MODELDIR, exp_fname)

    encoderfname = path.join(model_path, 'epoch%d.encoder.model' % epoch)
    decoderfname = path.join(model_path, 'epoch%d.decoder.model' % epoch)
    encoder.load_state_dict(torch.load(encoderfname))
    decoder.load_state_dict(torch.load(decoderfname))
