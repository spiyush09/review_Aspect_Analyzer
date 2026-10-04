"""
data_loader.py

Downloads the SemEval-2014 restaurant data, reads it into a table with one row
per (sentence, aspect) pair, and splits it into train / val / test.
"""
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_dataset():
    # download the XML file the first time
    if not config.DATA_FILE.exists():
        config.DATA_DIR.mkdir(exist_ok=True)
        print("Downloading dataset ...")
        urllib.request.urlretrieve(config.DATA_URL, config.DATA_FILE)

    rows = []
    for sent in ET.parse(config.DATA_FILE).getroot().iter("sentence"):
        terms = sent.find("aspectTerms")
        if terms is None:  # sentence without aspects
            continue
        for term in terms.findall("aspectTerm"):
            polarity = term.get("polarity")
            if polarity not in config.LABEL2ID:  # drop the "conflict" label
                continue
            rows.append({
                "sent_id": sent.get("id"),
                "sentence": sent.find("text").text,
                "aspect": term.get("term"),
                "sentiment": polarity,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["sent_id", "aspect"])
    df["label"] = df["sentiment"].map(config.LABEL2ID)
    return df.reset_index(drop=True)


def split_data(df):
    """
    Split into train / val / test by SENTENCE (not by row), so the same
    sentence never appears in both train and test.
    """
    sent_ids = df["sent_id"].unique()
    holdout = config.VAL_SIZE + config.TEST_SIZE

    train_ids, rest_ids = train_test_split(
        sent_ids, test_size=holdout, random_state=config.SEED)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=config.TEST_SIZE / holdout, random_state=config.SEED)

    train = df[df["sent_id"].isin(train_ids)].reset_index(drop=True)
    val = df[df["sent_id"].isin(val_ids)].reset_index(drop=True)
    test = df[df["sent_id"].isin(test_ids)].reset_index(drop=True)
    return train, val, test
