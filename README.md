# Customer Review Aspect Analyzer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spiyush09/review_Aspect_Analyzer/blob/main/notebooks/train_colab.ipynb)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hugging%20Face-yellow)](https://huggingface.co/pksingh09/review-aspect-analyzer-bert)

A web app that reads a customer review, finds the things people talk about (like food, service or price), and tells whether the opinion on each one is **positive**, **negative** or **neutral**.

> "The battery life is amazing, but the camera is terrible and the price is too high."

| Aspect  | Sentiment |
| ------- | --------- |
| battery | positive  |
| camera  | negative  |
| price   | negative  |

A normal sentiment classifier gives **one** label for a whole review ("mixed"), which is not useful for a company. This project gives a **separate sentiment for each aspect**.

---

## Choose your path

| | Option A: Use the pre-trained model | Option B: Train it yourself |
| --- | --- | --- |
| Best for | Trying the app quickly | Reproducing the results |
| Time | About 5 minutes | About 15 minutes (free Colab GPU) |
| How | Download the model from Hugging Face | Run the Colab notebook |

Both options share the same setup (steps 1-3 below), then differ only in **step 4**.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Quick start](#quick-start)
3. [Using the app](#using-the-app)
4. [Dataset](#dataset)
5. [Project structure](#project-structure)
6. [Results](#results)
7. [Limitations](#limitations)
8. [Future work](#future-work)
9. [Troubleshooting](#troubleshooting)
10. [Tech used](#tech-used)

---

## How it works

```
review -> split into sentences -> find aspects (spaCy) -> sentiment per aspect (BERT) -> result table
```

1. **Aspect extraction:** spaCy noun phrases. The head noun of each phrase (lemmatized) is a candidate aspect. This step is rule based.
2. **Aspect sentiment classification:** a pretrained BERT model fine-tuned (transfer learning) on SemEval-2014. The sentence and the aspect are given to the model as a *pair*, and it predicts negative / neutral / positive.
3. **Interface:** a Streamlit app where you type a review (or pick an example) and see the sentiment of each aspect.

---

## Quick start

**Requirements:** Python 3.9+ and git. Linux is the primary platform; Windows is also supported.

### 1. Clone the repository

```bash
git clone https://github.com/spiyush09/review_Aspect_Analyzer.git
cd review_Aspect_Analyzer
```

### 2. Create and activate a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

<details>
<summary>PowerShell blocks the activation script?</summary>

Use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

</details>

Your prompt should now start with `(.venv)`.

### 3. Install dependencies

The commands are the same on Linux and Windows:

```bash
python -m pip install --upgrade pip

# CPU-only PyTorch (avoids downloading large CUDA/NVIDIA packages)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else
python -m pip install -r requirements.txt

# spaCy English model
python -m spacy download en_core_web_sm
```

> Have an NVIDIA GPU and want to use it locally? Skip the CPU-only line and install PyTorch from <https://pytorch.org/get-started/locally/> instead.

### 4. Get the model

#### Option A: Download the pre-trained model (recommended for a quick test)

The trained model is hosted on Hugging Face: <https://huggingface.co/pksingh09/review-aspect-analyzer-bert>

Run this from the project folder (same command on Linux and Windows):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('pksingh09/review-aspect-analyzer-bert', local_dir='models/absa_model')"
```

This downloads about 430 MB into `models/absa_model/`.

#### Option B: Train it yourself on Google Colab

Training on a CPU is slow, so use a free Colab GPU.

1. Click the **Open in Colab** badge at the top of this README (or open `notebooks/train_colab.ipynb`).
2. Go to **Runtime -> Change runtime type -> T4 GPU**.
3. Go to **Runtime -> Run all**. Training takes about 10-15 minutes.
4. The last cell downloads `output.zip`. Extract it **inside your project folder**:

   Linux:

   ```bash
   unzip output.zip
   ```

   Windows: right-click `output.zip` -> **Extract All** into the project folder.

Either way, you should now have:

```
review_Aspect_Analyzer/
└── models/
    └── absa_model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── tokenizer_config.json
```

Want to try another model (for example `roberta-base`)? Change `MODEL_NAME` in `config.py` and train again.

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501> in your browser.

---

## Using the app

1. Type or paste a customer review, or choose a ready-made example.
2. Click **Analyze review**.
3. See the extracted aspects, predicted sentiments, confidence scores, and the sentences they came from.

---

## Dataset

**SemEval-2014 Task 4** (aspect term polarity), restaurant reviews. The file is downloaded automatically by the code.

- Each row is one (sentence, aspect) pair with a sentiment label.
- The "conflict" label is removed, leaving 3 classes.
- Split: 80% train / 10% validation / 10% test.
- The split is done by **sentence** (not by row), so the same sentence never appears in both train and test.

Reference: Pontiki et al., *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*, 2014. <https://aclanthology.org/S14-2004>

---

## Project structure

```
review_Aspect_Analyzer/
├── config.py                # all settings (model, dataset, hyperparameters)
├── requirements.txt
├── models/                  # the trained model goes here
├── results/                 # accuracy, F1, confusion matrix (created by training)
├── notebooks/
│   └── train_colab.ipynb    # one-click training on Google Colab
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

---

## Results

Saved in `results/` after training:

- `metrics.json`: validation and test accuracy / macro-F1, plus training settings
- `classification_report.txt`: precision, recall and F1 per class on the test set
- `confusion_matrix.png`: confusion matrix on the test set

| Model             | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
| ----------------- | ------------ | ------------ | ------------- | ------------- |
| bert-base-uncased | 0.7657       | 0.6761       | 0.7726        | 0.7089        |

Training settings: 4 epochs, learning rate 2e-5, batch size 16, seed 42.

Accuracy is higher than macro-F1 because the data has many more positive examples than neutral ones, and the model finds the neutral class hardest.

---

## Limitations

- Aspect extraction is rule based, so it can miss aspects written as verbs or adjectives ("it overheats") and sometimes picks nouns that are not real aspects.
- The model is trained on restaurant reviews only, so it may not work as well on other products.
- It was trained on single sentences, so sarcasm and long-range context are hard.
- The "neutral" class is the hardest, because it has fewer examples.

---

## Future work

- Make the app download the model automatically on first run.
- Train a token classification model for the aspect extraction step.
- Group similar aspects together (for example "waiter" and "staff").
- Analyze many reviews at once (CSV upload) and show a summary per aspect.
- Collect real tweets or Reddit posts and analyze them.
- Test more models (roberta, deberta) and other domains (laptops).

---

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| App says "no trained model found" | Check that `models/absa_model/` exists and contains `config.json` and `model.safetensors`. See [step 4](#4-get-the-model). |
| `ModuleNotFoundError` | Your virtual environment is not active. Re-run the activation command from step 2. |
| spaCy error: `Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` inside the active environment. |
| PowerShell says running scripts is disabled | Use Command Prompt (`.venv\Scripts\activate.bat`) or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| `streamlit: command not found` | Use `python -m streamlit run app/streamlit_app.py`. |
| Model download is slow or fails | Check your internet connection and re-run the command; it resumes where it stopped. |
| Colab shows `CUDA available: False` | Set **Runtime -> Change runtime type -> T4 GPU** and run again. |

---

## Tech used

Python, PyTorch, Hugging Face Transformers, spaCy, scikit-learn, Streamlit.
