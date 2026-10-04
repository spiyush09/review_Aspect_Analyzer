"""
aspect_extractor.py

Finds the aspects (like "battery", "service", "price") in a review with spaCy:
split the review into sentences, then take the head noun of every noun phrase.
This part is rule based, so there is nothing to train.
"""
from functools import lru_cache

import spacy

import config


@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load(config.SPACY_MODEL)  # loaded only once


def _aspects_in_sentence(sent):
    """Candidate aspects (lemmatized, lowercase) from one spaCy sentence."""
    found = []
    for chunk in sent.noun_chunks:
        root = chunk.root
        if root.pos_ not in ("NOUN", "PROPN") or root.is_stop:
            continue
        aspect = root.lemma_.lower()
        if (len(aspect) < config.MIN_ASPECT_LENGTH
                or not aspect.isalpha()
                or aspect in config.IGNORE_ASPECTS
                or aspect in found):
            continue
        found.append(aspect)
    return found[:config.MAX_ASPECTS_PER_SENTENCE]


def extract_aspect_pairs(text):
    """
    Returns a list of (sentence, aspect) pairs for the whole review.
    We work sentence by sentence because the model was trained on single sentences.
    """
    pairs = []
    for sent in get_nlp()(str(text)).sents:
        for aspect in _aspects_in_sentence(sent):
            pairs.append((sent.text.strip(), aspect))
    return pairs
