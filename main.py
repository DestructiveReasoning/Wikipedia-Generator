from os import path

from vocab import loadvocab, loadembeddings


DATADIR = 'data/processed'
TESTSET = path.join(DATADIR, 'test.bin')

VOCABFNAME = path.join(DATADIR, 'vocab')
VOCABSIZE = 150000

GLOVEFNAME = 'resources/glove/glove.6B.50d.txt'


def indices_from_text(text, vocab):
    text = text.split(' ')
    return [vocab.word_to_index(w) for w in text]


if __name__ == '__main__':
    with open(VOCABFNAME, 'r') as f:
        vocab = loadvocab(f, VOCABSIZE)

    with open(GLOVEFNAME, 'r') as f:
        emb = loadembeddings(f, vocab)

    with open(TESTSET, 'r') as f:
        _ = f.readline()
        abstract = f.readline()
        print(indices_from_text(abstract, vocab))
