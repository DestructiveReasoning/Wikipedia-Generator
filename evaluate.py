from train import variableFromSentence


def getEvalData(f, vocab):
    articles = []
    abstracts = []
    articleVars = []
    for i, line in enumerate(f):
        if i % 2 == 0:
            articles.append(line)
            v = variableFromSentence(
                vocab,
                vocab.SOS + ' ' + line + ' ' + vocab.EOS)
            articleVars.append(v)
        else:
            abstracts.append(line)
    return list(zip(articles, abstracts, articleVars))
