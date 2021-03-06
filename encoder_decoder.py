"""Adapted from: pytorch's
Translation with a Sequence to Sequence Network and Attention tutorial
https://bit.ly/2quRugE
"""

import time
import math
import random

import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch import optim

from vocab import Vocab


MAX_LENGTH = 100

use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 1.00  # TODO: Change such that teacher force sometime


def make_pairs(articles, abstracts):
    return list(zip(articles, abstracts))


def variableFromSentence(vocab, sentence):
    indices = vocab.words_to_indices(sentence)
    result = Variable(torch.LongTensor(indices).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = embedding
        if embedding is None:
            self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(input.size()[0], 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        result = (Variable(torch.zeros(2, 1, self.hidden_size)),
                  Variable(torch.zeros(2, 1, self.hidden_size)))
        if use_cuda:
            return (result[0].cuda(), result[1].cuda())
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, vocab):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_variable.size()[0]

    loss = 0

    _, encoder_hidden = encoder(input_variable, encoder_hidden)

    SOS_token = vocab.word_to_index(Vocab.SOS)
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = (encoder_hidden[0].view(1, 1, -1),
                      encoder_hidden[1].view(1, 1, -1))

    use_teacher_forcing = False
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == Vocab.EOS:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


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


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def trainIters(encoder, decoder, n_iters, training_pairs, vocab,
               print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion,
                     vocab)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                  iter, iter / n_iters * 100, print_loss_avg))
            i = random.randrange(len(input_variable))
            decoded = _evaluate(encoder, decoder, input_variable[i], vocab,
                                100)
            print(' '.join(decoded))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


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
