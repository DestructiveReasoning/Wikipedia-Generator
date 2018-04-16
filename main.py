from os import path

import torch

from vocab import loadvocab, loadembeddings
from encoder_decoder import EncoderRNN, DecoderRNN, variableFromSentence, \
        make_pairs, trainIters, evaluate


DATADIR = 'data/processed'
TRAINSET = path.join(DATADIR, 'train.bin')
TESTSET = path.join(DATADIR, 'test.bin')

VOCABFNAME = path.join(DATADIR, 'vocab')
VOCABSIZE = 100000

GLOVEFNAME = 'resources/glove/glove.6B.50d.txt'

HIDDEN_SIZE = 50
N_ITERS = 75000
EPOCHS = 5

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    with open(VOCABFNAME, 'r') as f:
        vocab = loadvocab(f, VOCABSIZE)

    with open(GLOVEFNAME, 'r') as f:
        emb = loadembeddings(f, vocab)

    articles = []
    abstracts = []
    with open(TRAINSET, 'r') as f:
        for i in range(N_ITERS):
            articles.append(f.readline())
            abstracts.append(f.readline())

    # TODO: add batching so we don't have load this whole thing into memory
    art_inputs = list(map(lambda x: variableFromSentence(vocab, x), articles))
    abs_inputs = list(map(lambda x: variableFromSentence(vocab, x), abstracts))
    pairs = make_pairs(art_inputs, abs_inputs)

    # TODO: change dimesion of hidden layer
    encoder = EncoderRNN(vocab.size(), HIDDEN_SIZE, emb)
    decoder = DecoderRNN(2*HIDDEN_SIZE, vocab.size())
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Starting to train...")
    for i in range(EPOCHS):
        print("epoch %d / %d" % (i, EPOCHS))
        trainIters(encoder, decoder, N_ITERS, pairs, vocab, print_every=100,
                   plot_every=1000)
    print("Done training.")

    decoded_words = evaluate(encoder, decoder, articles[0], vocab)
    print(decoded_words)
