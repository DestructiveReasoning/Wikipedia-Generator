import numpy as np
import torch


VOCABFNAME = 'data/processed/vocab'
VOCABSIZE = 150000

GLOVEFNAME = 'resources/glove/glove.6B.50d.txt'

# Stanford NLP repaces left and right parenthesis with special chars
LEFTBRACE = '-lrb-'
RIGHTBRACE = '-rrb-'


class Vocab:
    UNK = '<unk>'
    UNK_i = 0

    def __init__(self, vocab):
        self.vocab = vocab

    def word_to_index(self, word):
        if word in self.vocab:
            return self.vocab[word] + 1
        return Vocab.UNK_i

    def size(self):
        return len(self.vocab) + 1  # plus one for <unk>


def loadvocab(f, limit=None):
    """Returns a dictionary with the index of each word in the vocab.

    Keyword arguments:
    f -- file object containing the vocab as [word, count] pairs
    limit -- the max size of the vocab. By default uses all vocab
    """
    vocab = dict()
    i = 0
    for l in f:
        if limit is not None and i >= limit:
            break
        w, _ = l.split(' ')

        # don't include numbers in the vocab
        if any(c.isdigit() for c in w):
            continue

        # replace special token
        if w == LEFTBRACE:
            w = '('
        elif w == RIGHTBRACE:
            w = ')'

        vocab[w] = i
        i += 1
    return Vocab(vocab)


def loadembeddings(f, vocab):
    """Returns indexed embeddings for a vocab.

    Keyword arguments:
    f -- file object containing the word vectors as [word, [vec]] pairs
    vocab -- a vocab dict with words as the keys and indices as the values
    """
    wordvectors = [None]*vocab.size()
    for l in f:
        vec = l.split(' ')
        w, vec = vec[0], vec[1:]
        if vocab.word_to_index(w) == Vocab.UNK_i:
            continue
        i = vocab.word_to_index(w)
        wordvectors[i] = np.array(vec, dtype=float)

    # initialize words without embeddings to random vectors
    M = vocab.size()
    N = len(wordvectors[1])
    wordvectors = list(map(lambda v: np.random.rand(N) if v is None else v,
                           wordvectors))

    embed = torch.nn.Embedding(M, N)
    embed.weight.data = torch.FloatTensor(wordvectors)
    return embed
