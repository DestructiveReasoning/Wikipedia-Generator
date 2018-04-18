import numpy as np
import re

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

def make_summary(story, target_length, n=1):
    probs = get_token_probs(story, n=n)
    sentences = get_sentences(story.lower())
    summary = ""
    summary_len = 0
    while summary_len < target_length:
        sentence_scores = [score_sentence(s, probs, n=n) for s in sentences]
        sentences = [x for _, x in sorted(zip(sentence_scores, sentences), reverse=True)]
        summary += sentences[0] + ". "
        new_words = get_words(sentences[0])
        new_tokens = get_ngrams(sentences[0],n=n)
        summary_len += len(new_words)
        for token in new_tokens:
            probs[token] *= probs[token]

    return summary
