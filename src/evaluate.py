"""
evaluate.py

Tests the trained model on the test split and saves the classification report
and the confusion matrix in the results/ folder. Called by train.py.
"""
import matplotlib
matplotlib.use("Agg")  # no display needed (works in Colab and on servers)
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import config
from src.predict import AspectSentimentPredictor


def evaluate_model(test_df):
    predictor = AspectSentimentPredictor(config.MODEL_DIR)
    preds = predictor.predict(list(zip(test_df["sentence"], test_df["aspect"])))
    y_pred = [label for label, _ in preds]
    y_true = test_df["sentiment"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=config.LABELS, digits=3)
    print(f"Test accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(report)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "classification_report.txt").write_text(report)

    # confusion matrix picture
    cm = confusion_matrix(y_true, y_pred, labels=config.LABELS)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), config.LABELS)
    ax.set_yticks(range(3), config.LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return acc, macro_f1
