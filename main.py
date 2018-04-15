from os import path

import torch

from vocab import loadvocab, loadembeddings
from encoder_decoder import EncoderRNN, DecoderRNN, variableFromSentence, \
        make_pairs, trainIters, evaluate


DATADIR = 'data/processed'
TESTSET = path.join(DATADIR, 'test.bin')

VOCABFNAME = path.join(DATADIR, 'vocab')
VOCABSIZE = 50000

GLOVEFNAME = 'resources/glove/glove.6B.50d.txt'

HIDDEN_SIZE = 50
N_ITERS = 1000

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    with open(VOCABFNAME, 'r') as f:
        vocab = loadvocab(f, VOCABSIZE)

    with open(GLOVEFNAME, 'r') as f:
        emb = loadembeddings(f, vocab)

    articles = []
    abstracts = []
    with open(TESTSET, 'r') as f:
        for i in range(N_ITERS):
            articles.append(f.readline())
            abstracts.append(f.readline())

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
    trainIters(encoder, decoder, N_ITERS, pairs, vocab, print_every=10,
               plot_every=1000)
    print("Done training.")

    decoded_words = evaluate(encoder, decoder, articles[0], vocab)
    print(decoded_words)
