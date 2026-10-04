# Customer Review Aspect Analyzer

A web app that reads a customer review, finds the things people talk about (like food, service or price), and tells whether the opinion on each one is positive, negative or neutral.

---

## 1. About the project

A normal sentiment classifier gives **one** label for a whole review. But real
reviews are mixed:

> "The battery life is amazing, but the camera is terrible and the price is too high."

Overall this is "mixed", which is not useful for a company. This project finds
the **aspects** that are mentioned and gives a **separate sentiment for each one**:

| Aspect  | Sentiment |
|---------|-----------|
| battery | positive  |
| camera  | negative  |
| price   | negative  |

## 2. How it works

```
review -> split into sentences -> find aspects (spaCy) -> sentiment per aspect (BERT) -> result table
```

1. **Aspect extraction:** spaCy noun phrases. The head noun of each phrase
   (lemmatized) is taken as a candidate aspect. This part is rule based.
2. **Aspect sentiment classification:** a pretrained BERT model fine-tuned
   (transfer learning) on SemEval-2014. The sentence and the aspect are given
   to the model as a *pair*, and it predicts negative / neutral / positive.
3. **Interface:** a Streamlit app where you type a review (or pick a ready-made example) and see the sentiment of each aspect.

## 3. Dataset

**SemEval-2014 Task 4** (aspect term polarity), restaurant reviews.
The file is downloaded automatically by the code.

- Each row = one (sentence, aspect) pair with a sentiment label.
- The "conflict" label is removed, so we have 3 classes.
- Split: 80% train / 10% validation / 10% test.
- We split by **sentence** (not by row) so that the same sentence never appears
  in both train and test.

Reference: Pontiki et al., *SemEval-2014 Task 4: Aspect Based Sentiment
Analysis*, 2014. https://aclanthology.org/S14-2004

## 4. Project structure

```
review_Aspect_Analyzer/
├── config.py                # all settings (model, dataset, hyperparameters)
├── requirements.txt
├── models/                  # the trained model goes here
├── results/                 # accuracy, F1, confusion matrix (created by training)
├── src/
│   ├── data_loader.py       # download, read and split the data
│   ├── train.py             # fine-tune BERT
│   ├── evaluate.py          # test metrics + confusion matrix
│   ├── predict.py           # load the model and predict
│   ├── aspect_extractor.py  # spaCy aspect extraction
│   └── aggregate.py         # extract aspects + predict sentiment for a review
└── app/
    └── streamlit_app.py     # web page
```

## 5. Setup

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 6. How to run

### Step 1: train the model (use Google Colab with a GPU)

Training on CPU is slow, so we do this step in Colab
(Runtime -> Change runtime type -> T4 GPU). In a Colab notebook:

```python
!git clone https://github.com/spiyush09/review_Aspect_Analyzer.git
%cd review_Aspect_Analyzer
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm
!python -m src.train
```

This takes a few minutes. The model is saved in `models/absa_model/` and the
results in `results/`. Download both:

```python
!zip -r output.zip models/ results/
from google.colab import files
files.download("output.zip")
```

Unzip it inside the project folder on your computer, so that you have
`models/absa_model/` and `results/`.

### Step 2: run the app

```bash
streamlit run app/streamlit_app.py
```

Type a review, or pick one of the example reviews from the dropdown, and click
**Analyze review**.

To try another model (for example `roberta-base`), change `MODEL_NAME` in
`config.py` and train again.

## 7. Results

Saved in the `results/` folder after training:

- `run_info.json`: validation and test accuracy / macro-F1, and the training settings
- `confusion_matrix.png`: confusion matrix on the test set

| Model              | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|--------------------|--------------|--------------|---------------|---------------|
| bert-base-uncased  | 0.7711       | 0.6671       | 0.8027        | 0.7408        |

Training settings: 4 epochs, learning rate 2e-5, batch size 16, seed 42.

Accuracy is higher than macro-F1 because the data has many more positive
examples than neutral ones, and the model finds the neutral class hardest.

## 8. Limitations

- Aspect extraction is rule based, so it can miss aspects written as verbs or
  adjectives ("it overheats") and sometimes picks nouns that are not real aspects.
- The model is trained on restaurant reviews only, so it
  may not work as well on other products.
- It was trained on single sentences, so sarcasm and long-range context are hard.
- The "neutral" class is the hardest one for the model, because it has fewer examples.

## 9. Future work

- Train a token classification model for the aspect extraction step.
- Group similar aspects together (for example "waiter" and "staff").
- Analyze many reviews at once (CSV upload) and show a summary per aspect.
- Collect real tweets or Reddit posts and analyze them.
- Test more models (roberta, deberta) and other domains (laptops).

## 10. Tech used

Python, PyTorch, Hugging Face Transformers, spaCy, scikit-learn, Streamlit.