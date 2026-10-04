"""
aggregate.py

Runs the full pipeline on one review: extract aspects (spaCy) -> predict the
sentiment of every aspect (BERT) -> return a table.
"""
import pandas as pd

from src.aspect_extractor import extract_aspect_pairs


def analyze_review(review, predictor):
    pairs = extract_aspect_pairs(review)
    predictions = predictor.predict(pairs) if pairs else []

    rows = [{"aspect": aspect, "sentiment": label,
             "confidence": round(score, 3), "sentence": sentence}
            for (sentence, aspect), (label, score) in zip(pairs, predictions)]
    return pd.DataFrame(rows, columns=["aspect", "sentiment", "confidence", "sentence"])
