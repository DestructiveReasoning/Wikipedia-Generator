"""Adapted from: pytorch's
Translation with a Sequence to Sequence Network and Attention tutorial
https://bit.ly/2quRugE
"""

import time
import math
import random
import logging

import torch
from torch.autograd import Variable

from vocab import Vocab


use_cuda = torch.cuda.is_available()


def variableFromSentence(vocab, sentence, padded_length=None):
    indices = vocab.words_to_indices(sentence)
    result = Variable(torch.LongTensor(indices).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def getArticleAbstractPairs(f, vocab):
    articles = []
    abstracts = []
    for i, line in enumerate(f):
        line = variableFromSentence(vocab, line + ' ' + vocab.EOS)
        if i % 2 == 0:
            articles.append(line)
        else:
            abstracts.append(line)
    return list(zip(articles, abstracts))


class Trainer:
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 criterion, teacher_forcing_ratio, vocab):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.vocab = vocab

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

    def _non_teacher_forcing(self, decoder_hidden, decoder_input,
                             target_variable):
        # Without teacher forcing: use its own predictions as the next input
        loss = 0
        target_length = target_variable.size()[0]
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += self.criterion(decoder_output, target_variable[di])
            if ni == Vocab.EOS:
                break
        return loss

    def train(self, input_variable, target_variable):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        _, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        SOS_token = self.vocab.word_to_index(Vocab.SOS)
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = (encoder_hidden[0].view(1, 1, -1),
                          encoder_hidden[1].view(1, 1, -1))

        use_teacher_forcing = False
        if random.random() < self.teacher_forcing_ratio:
            use_teacher_forcing = True

        # Teacher forcing uses the last input from the target instead of the
        # encoder generated value
        if use_teacher_forcing:
            loss = self._teacher_forcing(decoder_hidden, decoder_input,
                                         target_variable)
        else:
            loss = self._non_teacher_forcing(decoder_hidden, decoder_input,
                                             target_variable)

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        target_length = target_variable.size()[0]
        return loss.data[0] / target_length

    def trainAll(self, training_pairs, print_every=1000):
        start = time.time()
        print_loss_total = 0  # Reset every print_every

        n_iters = len(training_pairs)

        for iter, (input, target) in enumerate(training_pairs):
            loss = self.train(input, target)
            print_loss_total += loss

            if (iter + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' %
                      (timeSince(start, (iter + 1) / n_iters),
                       (iter + 1), (iter + 1) / n_iters * 100,
                       print_loss_avg))
                logging.info('%.4f' % print_loss_avg)
                i = random.randrange(len(input))
                decoded = _evaluate(self.encoder, self.decoder, input[i],
                                    self.vocab, 100)
                print(' '.join(decoded))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluate(encoder, decoder, sequence, vocab, max_summary_length=100):
    input_variable = variableFromSentence(vocab, sequence)
    return _evaluate(encoder, decoder, input_variable, vocab,
                     max_summary_length)


def _evaluate(encoder, decoder, input_variable, vocab, max_summary_length=100):
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    for i in range(input_length):
        _, encoder_hidden = encoder(input_variable[i], encoder_hidden)

    SOS_token = vocab.word_to_index(Vocab.SOS)
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    if use_cuda:
        decoder_input = decoder_input.cuda()

    decoder_hidden = (encoder_hidden[0].view(1, 1, -1),
                      encoder_hidden[1].view(1, 1, -1))

    decoded_words = []

    for i in range(max_summary_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == vocab.word_to_index(Vocab.EOS):
            decoded_words.append("<EOS>")
            break
        else:
            decoded_words.append(vocab.index_to_word(ni))

        decoder_input = Variable(torch.LongTensor([[ni]]))
        if use_cuda:
            decoder_input = decoder_input.cuda()
    return decoded_words
