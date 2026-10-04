# Customer Review Aspect Analyzer: Project Documentation

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement and Objectives](#2-problem-statement-and-objectives)
3. [System Overview](#3-system-overview)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Project Structure](#6-project-structure)
7. [Code Explanation](#7-code-explanation)
8. [Results and Evaluation](#8-results-and-evaluation)
9. [Installation and Usage](#9-installation-and-usage)
10. [Limitations](#10-limitations)
11. [Future Work](#11-future-work)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Introduction

Online reviews are one of the main ways customers share their opinions about a product or a service. Companies read these reviews to understand what people like and what needs to be improved. The problem is that a single review usually talks about many things at once. For example:

> "The food was amazing but the service was really slow and the prices are too high."

If we use a normal sentiment classifier, it will give one label for the whole sentence (positive, negative or mixed). That does not tell the restaurant owner *what* went well and *what* went wrong.

This project solves that using **Aspect-Based Sentiment Analysis (ABSA)**. It first finds the **aspects** (the things people talk about, like food, service or price) and then predicts the **sentiment of each aspect separately**.

---

## 2. Problem Statement and Objectives

**Problem statement:** Given a customer review, find the aspects mentioned in it and classify the sentiment towards each aspect as positive, negative or neutral.

**Objectives:**

- Extract the aspects from a review automatically.
- Fine-tune a pretrained BERT model to predict the sentiment for a given (sentence, aspect) pair.
- Evaluate the model properly on data it has not seen during training.
- Build a simple web page where anyone can test the model.

---

## 3. System Overview

The whole system works as a pipeline:

```
Review text
    |
    v
Split into sentences            (spaCy)
    |
    v
Find aspects in each sentence   (spaCy noun phrases)
    |
    v
(sentence, aspect) pairs
    |
    v
Sentiment for each pair         (fine-tuned BERT)
    |
    v
Result table in the web app     (Streamlit)
```

There are two parts to the project:

1. **Training part (offline):** we download the dataset, fine-tune BERT, test it, and save the model and the results. This is done once, on Google Colab with a GPU.
2. **App part (online):** the Streamlit page loads the saved model, takes a review from the user, and shows the aspect-wise sentiment.

---

## 4. Dataset

We used **SemEval-2014 Task 4** (aspect term polarity), restaurant reviews. It is a well known benchmark dataset for ABSA. The code downloads the XML file automatically the first time it is run.

Each row in the dataset is one **(sentence, aspect, sentiment)** triple. A sentence with three aspects gives three rows.

| Item | Value |
|------|-------|
| Total (sentence, aspect) pairs | 3569 |
| Positive | 2151 |
| Negative | 789 |
| Neutral | 629 |

**Preprocessing steps:**

- Sentences with no aspect terms are skipped.
- The dataset also has a fourth label called "conflict". We removed it to keep a clean 3-class problem.
- Duplicate (sentence, aspect) pairs are dropped.
- The labels are converted to numbers: negative = 0, neutral = 1, positive = 2.

**Train / validation / test split:**

| Split | Share | Rows |
|-------|-------|------|
| Train | 80% | 2837 |
| Validation | 10% | 367 |
| Test | 10% | 365 |

We split by **sentence** and not by row. If one sentence has 3 aspects and its rows go into different splits, the model would see part of the test sentence during training. This is called data leakage and it gives fake high accuracy. Splitting by sentence avoids it. A fixed seed (42) is used so the split is the same every time.

---

## 5. Methodology

### 5.1 Aspect extraction

Aspect extraction is **rule based** and uses spaCy (`en_core_web_sm`). Nothing is trained in this step.

1. The review is split into sentences.
2. For every sentence, spaCy finds the noun phrases (for example "the service" or "the prices").
3. The **head noun** of each phrase is taken as a candidate aspect, converted to its base form (lemma) and lowercase. So "prices" becomes "price".
4. Bad candidates are removed: stop words, very short words (less than 3 letters), words with numbers or symbols, duplicates, and very generic nouns like "thing", "time", "day", "lot".
5. At most 6 aspects are kept per sentence.

### 5.2 Aspect sentiment classification

For the sentiment, we use **BERT** (`bert-base-uncased`) and fine-tune it on the dataset. This is **transfer learning**: BERT is already pretrained on a huge amount of English text, so it understands language well. We only add a small classification layer on top with 3 outputs (negative, neutral, positive) and train it for our task.

The important idea is how we feed the input. We give BERT the sentence and the aspect as a **pair**:

```
[CLS] The food was amazing but the service was slow [SEP] service [SEP]
```

This tells the model which aspect to focus on. Without this, a sentence like the one above would get the same answer for both "food" and "service", which is wrong because they have opposite sentiments.

### 5.3 Training settings

| Setting | Value |
|---------|-------|
| Base model | bert-base-uncased |
| Epochs | 4 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 128 |
| Seed | 42 |
| Hardware | Google Colab GPU |

After each epoch the model is checked on the validation set, and the **best epoch** (by macro-F1) is kept at the end. This stops us from saving an over-trained model.

### 5.4 Evaluation metrics

- **Accuracy:** the share of predictions that are correct.
- **Macro-F1:** the F1 score is calculated for each class and then averaged, giving every class equal weight. This is important here because the classes are not balanced (there are far fewer neutral examples). A model can have high accuracy just by predicting "positive" most of the time, but macro-F1 would show that weakness.

---

## 6. Project Structure

```
review_Aspect_Analyzer/
├── config.py                # all settings in one place
├── requirements.txt         # python packages needed
├── README.md
├── models/
│   └── absa_model/          # trained model (not pushed to GitHub, too big)
├── results/
│   ├── confusion_matrix.png # confusion matrix on the test set
│   └── run_info.json        # training settings and accuracy / F1 numbers
├── src/
│   ├── data_loader.py       # download, read and split the data
│   ├── train.py             # fine-tune BERT
│   ├── evaluate.py          # test the model, save report and matrix
│   ├── predict.py           # load the model and make predictions
│   ├── aspect_extractor.py  # find aspects using spaCy
│   └── aggregate.py         # full pipeline for one review
└── app/
    └── streamlit_app.py     # the web page
```

---

## 7. Code Explanation

### `config.py`

Keeps all the settings in one place: folder paths, dataset link, label names, split sizes, model name and training values, and the spaCy settings. If we want to try another model (for example `roberta-base`), we only change `MODEL_NAME` here.

### `src/data_loader.py`

- `load_dataset()` downloads the XML file if it is not there, reads every sentence and its aspect terms, removes the "conflict" label and duplicates, and returns a pandas table.
- `split_data(df)` splits the table into train, validation and test by sentence ID.

### `src/train.py`

The main training script. It loads and splits the data, tokenizes the (sentence, aspect) pairs, loads BERT with a 3-class head, and trains using the Hugging Face `Trainer`. After training it saves the model and tokenizer in `models/absa_model/`, deletes the large temporary checkpoint folder, and calls the evaluation on the test set. Run it with `python -m src.train`.

### `src/evaluate.py`

Loads the saved model, predicts on the test set, and calculates accuracy and macro-F1. It also saves a classification report (precision, recall and F1 for each class) and a confusion matrix image in the `results/` folder.

### `src/predict.py`

Contains the `AspectSentimentPredictor` class. It loads the model and tokenizer, and uses a GPU if one is available, otherwise the CPU. Its `predict()` method takes a list of (sentence, aspect) pairs, runs them through the model in batches, applies softmax, and returns the predicted label with its confidence score.

### `src/aspect_extractor.py`

Implements the rule based aspect extraction described in section 5.1. The main function `extract_aspect_pairs(text)` returns a list of (sentence, aspect) pairs for a whole review. The spaCy model is loaded only once and reused.

### `src/aggregate.py`

Connects the two parts. `analyze_review(review, predictor)` extracts the aspects, gets the sentiment for each pair, and returns a table with the columns: aspect, sentiment, confidence and sentence.

### `app/streamlit_app.py`

The web page. It loads the model once (cached), shows a dropdown of example reviews and a text box, and when the user clicks **Analyze review** it runs the pipeline and shows the result table with colour coded sentiments. If the text box is empty or no aspect is found, it shows a helpful message. If the trained model is missing, it tells the user to train one first.

---

## 8. Results and Evaluation

The model was evaluated on the validation set (used while training) and on the test set (never seen during training).

| Set | Accuracy | Macro-F1 |
|-----|----------|----------|
| Validation | 0.7711 | 0.6671 |
| **Test** | **0.8027** | **0.7408** |

The saved files in the `results/` folder are:

- `confusion_matrix.png`: shows for each true class how many predictions went to each class.
- `run_info.json`: the training settings and the numbers above.

(If the model is trained again with the current code, it will also save `metrics.json` and `classification_report.txt` in the same folder.)

**Observations:**

- The model reaches about **80% accuracy** on unseen test data, which is a good result for a 3-class aspect level task with a small dataset.
- Macro-F1 (0.74) is lower than accuracy (0.80). The reason is class imbalance: there are many more positive examples than neutral ones, so the model does best on positive and struggles most with **neutral**. Neutral opinions are naturally harder, because they have no strong sentiment words.

---

## 9. Installation and Usage

Python 3.9 or newer is recommended.

### 9.1 Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# CPU-only PyTorch (skip this line if you have an NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 9.2 Training the model

Training on a CPU is very slow, so Google Colab with a GPU (T4) is recommended:

```python
!git clone https://github.com/spiyush09/review_Aspect_Analyzer.git
%cd review_Aspect_Analyzer
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm
!python -m src.train
```

Then download the output and put it in the project folder:

```python
!zip -r output.zip models/ results/
from google.colab import files
files.download("output.zip")
```

The model must end up at `models/absa_model/`.

### 9.3 Running the web app

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in the browser, then:

1. Choose an example review from the dropdown, or type your own review.
2. Click **Analyze review**.
3. Read the table: each row is one aspect with its sentiment, confidence and the sentence it came from.

> **Note:** While running, the terminal may show warnings about `torchvision` not found. These come from Streamlit scanning the `transformers` library. They are harmless and do not affect the app.

---

## 10. Limitations

- **Rule based aspects:** the noun based extraction can miss aspects written as verbs or adjectives (for example "it overheats"), and sometimes picks nouns that are not real aspects.
- **Single domain:** the model is trained only on restaurant reviews, so it may not work as well for other products like phones or laptops.
- **Sentence level:** it was trained on single sentences, so sarcasm and meaning that depends on earlier sentences are hard for it.
- **Neutral class:** this is the weakest class because it has the fewest examples.
- **Small dataset:** about 3500 training pairs is small, so results can change with a different split.

---

## 11. Future Work

- Train a token classification model to extract aspects instead of using rules.
- Group similar aspects together (for example "waiter" and "staff").
- Allow analysing many reviews at once (CSV upload) and show a summary for each aspect.
- Train on more domains (laptops, hotels) and on real reviews from sources like Reddit or Twitter.
- Try other models such as RoBERTa or DeBERTa and compare the results.

---

## 12. Conclusion

This project shows that aspect-based sentiment analysis gives much more useful information than one overall sentiment label. By combining a rule based aspect extractor (spaCy) with a fine-tuned BERT classifier, the system finds what customers are talking about and how they feel about each thing. The model reached **80.27% accuracy and 0.74 macro-F1** on the test set, and the Streamlit app makes it easy for anyone to try it. The project also helped in understanding transfer learning, careful data splitting, and how to evaluate a model on imbalanced classes.

---

## 13. References

1. Pontiki, M., et al. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.* https://aclanthology.org/S14-2004
2. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* https://arxiv.org/abs/1810.04805
3. Hugging Face Transformers documentation. https://huggingface.co/docs/transformers
4. spaCy documentation. https://spacy.io
5. Streamlit documentation. https://docs.streamlit.io