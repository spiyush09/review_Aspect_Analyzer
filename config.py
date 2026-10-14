"""
config.py

All the settings of the project in one place.
"""
from pathlib import Path

# folders
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "absa_model"   # the trained model is saved here
RESULTS_DIR = ROOT / "results"               # accuracy, F1, confusion matrix

# dataset: SemEval-2014 Task 4, restaurant reviews (downloaded automatically)
DATA_FILE = DATA_DIR / "Restaurants_Train_v2.xml"
DATA_URL = ("https://raw.githubusercontent.com/davidsbatista/"
            "Aspect-Based-Sentiment-Analysis/master/datasets/ABSA-SemEval2014/"
            "Restaurants_Train_v2.xml")

# the 3 sentiment classes
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# train / validation / test split
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 42

# model and training (any Hugging Face model name works, e.g. "roberta-base")
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 16

# aspect extraction (spaCy)
SPACY_MODEL = "en_core_web_sm"
MAX_ASPECTS_PER_SENTENCE = 6
MIN_ASPECT_LENGTH = 3

# very generic nouns that are not useful as aspects
IGNORE_ASPECTS = {
    "thing", "things", "time", "day", "lot", "way", "bit", "kind",
    "one", "people", "everything", "something", "anything", "nothing",
    "year", "week", "month", "minute", "hour", "reason", "fact",
    "point", "end",

    # Generic review words that are usually not useful aspects
    "experience", "improvement", "room", "restaurant", "place",
    "visit", "overall", "result", "problem", "issue", "part",
    "lot", "someone", "everyone", "customer", "review"
}

