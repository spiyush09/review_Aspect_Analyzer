"""
train.py

Fine-tunes a pretrained transformer (BERT) for aspect-based sentiment
classification. Best run on a GPU (Google Colab):

    python -m src.train

The model is saved in models/, the accuracy / F1 / confusion matrix in results/.
"""
import json
import shutil

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import config
from src.data_loader import load_dataset, split_data
from src.evaluate import evaluate_model


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def main():
    # 1. data
    train_df, val_df, test_df = split_data(load_dataset())
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # 2. tokenizer: sentence and aspect are given as a PAIR, so the model knows
    #    which aspect we are asking about
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    def to_dataset(d):
        ds = Dataset.from_pandas(d[["sentence", "aspect", "label"]])
        return ds.map(lambda b: tokenizer(b["sentence"], b["aspect"],
                                          truncation=True, max_length=config.MAX_LEN),
                      batched=True)

    train_ds, val_ds = to_dataset(train_df), to_dataset(val_df)

    # 3. model (pretrained body + new 3-class head)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=len(config.LABELS),
        id2label=config.ID2LABEL, label2id=config.LABEL2ID)

    training_args = TrainingArguments(
        output_dir=str(config.MODEL_DIR / "checkpoints"),
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=50,
        seed=config.SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 4. train and save the model
    trainer.train()
    val_metrics = trainer.evaluate()
    trainer.save_model(str(config.MODEL_DIR))
    tokenizer.save_pretrained(str(config.MODEL_DIR))
    shutil.rmtree(config.MODEL_DIR / "checkpoints", ignore_errors=True)  # not needed anymore

    # 5. test on data the model never saw, then save all the numbers in results/
    test_acc, test_f1 = evaluate_model(test_df)
    metrics = {
        "model": config.MODEL_NAME,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "val_accuracy": round(val_metrics["eval_accuracy"], 4),
        "val_macro_f1": round(val_metrics["eval_f1"], 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone. Model saved in {config.MODEL_DIR}, results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
