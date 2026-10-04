"""
predict.py

Loads the trained model and predicts the sentiment for (sentence, aspect) pairs.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


class AspectSentimentPredictor:
    def __init__(self, model_dir, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, pairs):
        """
        pairs: list of (sentence, aspect) tuples
        returns: list of (label, confidence) in the same order
        """
        results = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tokenizer(
                [sentence for sentence, _ in batch],
                [aspect for _, aspect in batch],
                truncation=True,
                max_length=config.MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu()
            for p in probs:
                idx = int(p.argmax())
                results.append((self.model.config.id2label[idx], float(p[idx])))
        return results
