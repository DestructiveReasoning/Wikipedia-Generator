import sys
import json
import random
from os import path

import torch
from torch import nn
from torch.autograd import Variable

from main import DATADIR, MODELDIR, VOCABFNAME, HIDDEN_SIZE
from encoder_decoder import EncoderRNN, DecoderRNN
from vocab import loadvocab, Vocab
from train import variableFromSentence


VALSET = path.join(DATADIR, 'val.bin')

use_cuda = torch.cuda.is_available()


def getEvalData(f, vocab, max_num):
    articles = []
    abstracts = []
    for i, line in enumerate(f):
        line = variableFromSentence(
            vocab,
            line + ' ' + vocab.EOS)
        if i > max_num:
            break
        elif i % 2 == 0:
            articles.append(line)
        else:
            abstracts.append(line)
    return list(zip(articles, abstracts))


class Validater:
    def __init__(self, encoder, decoder, vocab, criterion, teacher_forcing_ratio):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.criterion = criterion
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def validate(self, input_variable, target):
        encoder_hidden = self.encoder.initHidden()

        _, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        SOS_token = self.vocab.word_to_index(vocab.SOS)
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = (encoder_hidden[0].view(1, 1, -1),
                          encoder_hidden[1].view(1, 1, -1))

        if random.random() < self.teacher_forcing_ratio:
            loss = self._teacher_forcing(decoder_hidden, decoder_input, target)
        else:
            loss = self._non_teacher_forcing(decoder_hidden, decoder_input, target)

        target_length = target.size()[0]
        return loss.data[0] / target_length

    def _non_teacher_forcing(self, decoder_hidden, decoder_input, target):
        # Without teacher forcing: use its own predictions as the next input
        loss = 0
        target_length = target.size()[0]
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += self.criterion(decoder_output, target[di])
            if ni == Vocab.EOS:
                break
        return loss

    def _teacher_forcing(self, decoder_hidden, decoder_input, target_variable):
        # Teacher forcing: Feed the target as the next input
        loss = 0
        target_length = target_variable.size()[0]
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
            loss += self.criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
        return loss

    def validateAll(self, data):
        loss = 0
        for art, abs in data:
            loss += self.validate(art, abs)
        return loss / len(data)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: puth load_model.py experiment.json")
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

    print("Loading Validation Set...")
    with open(VALSET, 'r') as f:
        valdata = getEvalData(f, vocab, 10000)

    encoder = EncoderRNN(vocab.size(), HIDDEN_SIZE)
    decoder = DecoderRNN(2*HIDDEN_SIZE, vocab.size())
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    exp_fname = path.split(EXPERIMENT)[-1].split('.')[0]
    model_path = path.join(MODELDIR, exp_fname)

    crit = nn.NLLLoss()

    for epoch in range(5):
        encoderfname = path.join(model_path, 'epoch%d.encoder.model' % epoch)
        decoderfname = path.join(model_path, 'epoch%d.decoder.model' % epoch)
        encoder.load_state_dict(torch.load(encoderfname))
        decoder.load_state_dict(torch.load(decoderfname))

        validater = Validater(encoder, decoder, vocab, crit, experiment['teacher_forcing'])
        print(validater.validateAll(valdata))
