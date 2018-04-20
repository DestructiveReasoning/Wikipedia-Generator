from sumbasic import get_words, get_sentences

def make_summary(story, n):
    sentences = get_sentences(story.lower())
    return ". ".join(sentences[:n])
