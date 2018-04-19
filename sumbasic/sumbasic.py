import numpy as np
import re

def exp_distribution(lam, x):
    return lam * np.exp(x)

def get_words(s):
    return re.sub("[^\w]", " ", s).split()

def get_ngrams(s, n=1):
    words = get_words(s)
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ""
        for j in range(n):
            ngram += words[i + j] + " "
        ngrams.append(ngram)
    return ngrams

def get_token_probs(story, n=1):
    d = {}
    tokens = get_ngrams(story.lower(),n=n)
    N = 0
    for token in tokens:
        if token not in d:
            d[token] = 1.0
            N += 1.0
        else:
            d[token] += 1
    for key in d.keys():
        d[key] /= N
    return d

def get_sentences(story):
    story.replace(". ", ".")
    return story.split('.')

def score_sentence(sentence, probs, n=1):
#    words = get_words(sentence.lower())
    tokens = get_ngrams(sentence.lower(), n=n)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0
    total_prob = sum(map(lambda x: probs[x] if x in probs else 0.0, tokens))
    return total_prob/total_tokens

def pick_sentence(sentences, probs, n=1, lam=1.0):
    sentence_scores = [score_sentence(s, probs, n=n) for s in sentences]
    sentences = [x for _, x in sorted(zip(sentence_scores, sentences), reverse=True)]
    distribution = exp_distribution(lam, range(len(sentences)))
    i = np.random.rand()
    x = 0
    while distribution[x] > i:
        i += 1
        if i >= len(distribution):
            i = 0
            break
    return sentences[i]

def make_summary(story, target_length, n=1, lam=1.0):
    probs = get_token_probs(story, n=n)
    sentences = get_sentences(story.lower())
    summary = ""
    summary_len = 0
    while summary_len < target_length:
        sentence = pick_sentence(sentences, probs, n=n, lam=lam)
        summary += sentence + ". "
        new_words = get_words(sentence)
        new_tokens = get_ngrams(sentence,n=n)
        summary_len += len(new_words)
        for token in new_tokens:
            probs[token] *= probs[token]

    return summary
