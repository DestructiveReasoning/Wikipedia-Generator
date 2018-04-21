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

def get_best_token(tokens):
    highest_prob = 0
    token = ""
    for key in tokens.keys():
        if tokens[key] > highest_prob:
            token = key
            highest_prob = tokens[key]
    return token

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
    sentence_tokens = list(map(lambda x: get_ngrams(x, n), sentences))
    best_token = get_best_token(probs)
    potential_sentences = [sentences[i] for i in range(len(sentences)) if best_token in sentence_tokens[i]]
    if len(potential_sentences) == 0:
        probs[best_token] = 0
        return None
    sentence_scores = [score_sentence(s, probs, n=n) for s in potential_sentences]
    sentences = [x for _, x in sorted(zip(sentence_scores, potential_sentences), reverse=True)]
    distribution = exp_distribution(lam, range(len(potential_sentences)))
    i = np.random.rand()
    x = 0
    while distribution[x] < i:
        x += 1
        if x >= len(distribution):
            x = 0 #In the unlucky event that i > all values in distribution, just revert to best sentence
            break
    return potential_sentences[x]

def make_summary(story, target_length, n=1, lam=1.0):
    probs = get_token_probs(story, n=n)
    sentences = get_sentences(story.lower())
    summary = ""
    summary_len = 0
    while summary_len < target_length:
        sentence = pick_sentence(sentences, probs, n=n, lam=lam)
        if sentence is None:
            continue
        summary += sentence + ". "
        new_words = get_words(sentence)
        new_tokens = get_ngrams(sentence,n=n)
        summary_len += len(new_words)
        for token in new_tokens:
            probs[token] *= probs[token]

    return summary
