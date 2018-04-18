import numpy as np
import re

def get_words(s):
    return re.sub("[^\w]", " ", s).split()

def get_word_probs(story):
    d = {}
    words = get_words(story.lower())
    N = 0
    for word in words:
        if word not in d:
            d[word] = 1.0
            N += 1.0
        else:
            d[word] += 1
    for key in d.keys():
        d[key] /= N
    return d

def get_sentences(story):
    story.replace(". ", ".")
    return story.split('.')

def score_sentence(sentence, probs):
    words = get_words(sentence.lower())
    total_words = len(words)
    if total_words == 0:
        return 0
    total_prob = sum(map(lambda x: probs[x] if x in probs else 0.0, words))
    return total_prob/total_words

def make_summary(story, target_length):
    probs = get_word_probs(story)
    sentences = get_sentences(story.lower())
    summary = ""
    summary_len = 0
    while summary_len < target_length:
        sentence_scores = [score_sentence(s, probs) for s in sentences]
        sentences = [x for _, x in sorted(zip(sentence_scores, sentences), reverse=True)]
        summary += sentences[0] + ". "
        new_words = get_words(sentences[0])
        summary_len += len(new_words)
        for word in new_words:
            probs[word] *= probs[word]

    return summary
