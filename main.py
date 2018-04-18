import os
import sys
import json
import logging
from os import path

import torch
from torch import nn

from vocab import loadvocab, loadembeddings
from encoder_decoder import EncoderRNN, DecoderRNN
from train import Trainer, getArticleAbstractPairs
from experiments import createOptimizer

DATADIR = 'data/processed'
TRAINSET = path.join(DATADIR, 'train.bin')
TESTSET = path.join(DATADIR, 'test.bin')

MODELDIR = 'models'

VOCABFNAME = path.join(DATADIR, 'vocab')
GLOVEFNAME = 'resources/glove/glove.6B.50d.txt'

HIDDEN_SIZE = 50
EPOCHS = 5

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python main.py experiment.json")
        sys.exit(-1)

    EXPERIMENT = sys.argv[1]
    if not path.isfile(EXPERIMENT):
        print("%s does not exist" % EXPERIMENT)
        sys.exit(-1)

    with open(EXPERIMENT, 'r') as f:
        experiment = json.load(f)

    print("Loading Vocab...")
    with open(VOCABFNAME, 'r') as f:
        vocab = loadvocab(f, experiment['vocab_size'])

    print("Loading Embeddings...")
    with open(GLOVEFNAME, 'r') as f:
        emb = loadembeddings(f, vocab)

    # Build the input
    print("Loading Training Set...")
    with open(TRAINSET, 'r') as f:
        pairs = getArticleAbstractPairs(f, vocab)

    # Initialize encoder and decoder
    encoder = EncoderRNN(vocab.size(), HIDDEN_SIZE, emb)
    decoder = DecoderRNN(2*HIDDEN_SIZE, vocab.size())
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # Optimizers
    encoder_optim = createOptimizer(experiment['optimizer'],
                                    encoder.parameters())
    decoder_optim = createOptimizer(experiment['optimizer'],
                                    decoder.parameters())

    # Loss function
    crit = nn.CrossEntropyLoss()

    trainer = Trainer(encoder, decoder, encoder_optim, decoder_optim, crit,
                      experiment['teacher_forcing'], vocab)

    # Create directory for saving models
    exp_fname = path.split(EXPERIMENT)[-1].split('.')[0]
    model_path = path.join(MODELDIR, exp_fname)
    os.makedirs(model_path)

    # logging for training error
    training_error_log = path.join(model_path, 'training_error.log')
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO,
                        filename=training_error_log)

    print("Starting Training...")
    for i in range(EPOCHS):
        print("epoch %d / %d" % (i, EPOCHS))
        trainer.trainAll(pairs, 100)

        # Save the model for future eval
        encoderfname = path.join(model_path, 'epoch%d.encoder.model' % i)
        decoderfname = path.join(model_path, 'epoch%d.decoder.model' % i)
        torch.save(encoder.state_dict(), encoderfname)
        torch.save(decoder.state_dict(), decoderfname)
