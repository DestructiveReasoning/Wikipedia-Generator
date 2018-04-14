from os import path

import torch
from torch.autograd import Variable

from vocab import Vocab, loadvocab, loadembeddings
from encoder_decoder import EncoderRNN, DecoderRNN, variableFromSentence, make_pairs, trainIters, evaluate


DATADIR = 'data/processed'
TESTSET = path.join(DATADIR, 'test.bin')

VOCABFNAME = path.join(DATADIR, 'vocab')
VOCABSIZE = 150000

GLOVEFNAME = 'resources/glove/glove.6B.50d.txt'

HIDDEN_SIZE = 50
MAX_LENGTH = 10000

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    with open(VOCABFNAME, 'r') as f:
        vocab = loadvocab(f, VOCABSIZE)

    with open(GLOVEFNAME, 'r') as f:
        emb = loadembeddings(f, vocab)

    articles = []
    abstracts = []
    with open(TESTSET, 'r') as f:
        for i in range(5):
            articles.append(f.readline())
            abstracts.append(f.readline())

    art_inputs = list(map(lambda x: variableFromSentence(vocab, x), articles))
    abs_inputs = list(map(lambda x: variableFromSentence(vocab, x), abstracts))
    pairs = make_pairs(art_inputs, abs_inputs)

#     TODO: change dimesion of hidden layer
    encoder = EncoderRNN(vocab.size(), HIDDEN_SIZE, emb)
    decoder = DecoderRNN(HIDDEN_SIZE, vocab.size())

    trainIters(encoder, decoder, 10, pairs, print_every=1, plot_every=1000, max_length=MAX_LENGTH)
#
#    article_input = variableFromSentence(vocab, article)
#    abstract_input = variableFromSentence(vocab, abstract)
#
#    encoder_hidden = encoder.initHidden()
#    encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
#    for i in range(article_input.size()[0]):
#        encoder_output, encoder_hidden = encoder(article_input[i], encoder_hidden)
#        encoder_outputs[i] = encoder_output[0][0]
#    
#
#    SOS_token = vocab.word_to_index(Vocab.SOS)
#    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
#    decoder_hidden = encoder_hidden 
#
#    for i in range(abstract_input.size()[0]):
#        decoder_ouput, decoder_hidden = decoder(decoder_input, decoder_hidden)
#        decoder_input = abstract_input[i]
