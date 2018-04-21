import sys
import json
from os import path

import torch
from torch.autograd import Variable
import rouge

from main import DATADIR, MODELDIR, VOCABFNAME, HIDDEN_SIZE
from encoder_decoder import EncoderRNN, DecoderRNN
from train import variableFromSentence
from vocab import loadvocab


VALSET = path.join(DATADIR, 'val.bin')

use_cuda = torch.cuda.is_available()

def generate_abstract(encoder, decoder, input_var, vocab, max_length=100):
    encoder_hidden = encoder.initHidden()

    _, encoder_hidden = encoder(input_var, encoder_hidden)

    SOS_token = vocab.word_to_index(vocab.SOS)
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    if use_cuda:
        decoder_input = decoder_input.cuda()

    decoder_hidden = (encoder_hidden[0].view(1, 1, -1),
                      encoder_hidden[0].view(1, 1, -1))

    decoded_words = []

    for i in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        _, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == vocab.word_to_index(vocab.EOS):
            break
        else:
            decoded_words.append(vocab.index_to_word(ni))

        decoder_input = Variable(torch.LongTensor([[ni]]))
        if use_cuda:
            decoder_input = decoder_input.cuda()

    return ' '.join(decoded_words)
    

def getEvalData(f, vocab):
    abstracts = []
    articleVars = []
    for i, line in enumerate(f):
        if i > 100:  # TODO: Remove this break
            break
        if i % 2 == 0:
            v = variableFromSentence(
                vocab,
                line + ' ' + vocab.EOS)
            articleVars.append(v)
        else:
            abstracts.append(line)
    return list(zip(abstracts, articleVars))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rouge_score.py experiment.json")
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
        valdata = getEvalData(f, vocab)

    encoder = EncoderRNN(vocab.size(), HIDDEN_SIZE)
    decoder = DecoderRNN(2*HIDDEN_SIZE, vocab.size())
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    exp_fname = path.split(EXPERIMENT)[-1].split('.')[0]
    model_path = path.join(MODELDIR, exp_fname)

    encoderfname = path.join(model_path, 'epoch4.encoder.model')
    decoderfname = path.join(model_path, 'epoch4.decoder.model')
    encoder.load_state_dict(torch.load(encoderfname))
    decoder.load_state_dict(torch.load(decoderfname))

    generated_abstracts = [] 
    targets = []
    for abstract, articleVar in valdata:
        gen_abstract = generate_abstract(encoder, decoder, articleVar, vocab)
        generated_abstracts.append(gen_abstract)
        targets.append(abstract)

    rouge = rouge.Rouge()
    scores = rouge.get_scores(generated_abstracts, targets, avg=True)
    print(scores['rouge-1']['f'], scores['rouge-2']['f'])
