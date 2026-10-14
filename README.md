# Customer Review Aspect Analyzer

A web app that reads a customer review, finds the things people talk about (like food, service or price), and tells whether the opinion on each one is positive, negative or neutral.

---

## 1. About the project

A normal sentiment classifier gives **one** label for a whole review. But real reviews are mixed:

> "The battery life is amazing, but the camera is terrible and the price is too high."

Overall this is "mixed", which is not useful for a company. This project finds the **aspects** that are mentioned and gives a **separate sentiment for each one**:

| Aspect | Sentiment |
|---|---|
| battery | positive |
| camera | negative |
| price | negative |

## 2. How it works

```text
review -> split into sentences -> find aspects (spaCy) -> sentiment per aspect (BERT) -> result table
```

1. **Aspect extraction:** spaCy noun phrases. The head noun of each phrase (lemmatized) is taken as a candidate aspect. This part is rule based.
2. **Aspect sentiment classification:** a pretrained BERT model fine-tuned (transfer learning) on SemEval-2014. The sentence and the aspect are given to the model as a *pair*, and it predicts negative / neutral / positive.
3. **Interface:** a Streamlit app where you type a review (or pick a ready-made example) and see the sentiment of each aspect.

## 3. Dataset

**SemEval-2014 Task 4** (aspect term polarity), restaurant reviews. The file is downloaded automatically by the code.

- Each row = one (sentence, aspect) pair with a sentiment label.
- The "conflict" label is removed, so we have 3 classes.
- Split: 80% train / 10% validation / 10% test.
- We split by **sentence** (not by row) so that the same sentence never appears in both train and test.

Reference: Pontiki et al., *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*, 2014. https://aclanthology.org/S14-2004

## 4. Project structure

```text
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

Python 3.9 or newer is recommended. A virtual environment is recommended to keep project dependencies separate from system Python.

### Linux (Fedora and other distributions)

**Step 1: Open the project directory**

```bash
cd ~/Projects/review_Aspect_Analyzer
```

**Step 2: Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)`.

**Step 3: Install the requirements**

The `requirements.txt` file should contain the following dependencies, without `torch` or the PyTorch CPU index URL:

```text
transformers>=4.41
datasets
accelerate
scikit-learn
matplotlib
spacy>=3.5
pandas
streamlit>=1.30
```

Install them:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Step 4: Install CPU-only PyTorch**

Install PyTorch separately to avoid selecting a CUDA-enabled build and downloading its associated NVIDIA dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 5: Install the spaCy English model**

```bash
python -m spacy download en_core_web_sm
```

**Step 6: Verify the installation**

```bash
python -c "import torch, transformers, spacy, pandas, streamlit; print('All imports successful'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For a CPU-only installation, `CUDA available` should be `False`.

### Windows

Open PowerShell in the project directory.

**Step 1: Create and activate a virtual environment**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

**Step 2: Install the requirements**

Use the same `requirements.txt` shown above:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Step 3: Install CPU-only PyTorch**

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 4: Install the spaCy English model**

```powershell
python -m spacy download en_core_web_sm
```

**Step 5: Verify the installation**

```powershell
python -c "import torch, transformers, spacy, pandas, streamlit; print('All imports successful'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For a CPU-only installation, `CUDA available` should be `False`.

---

## 6. How to run

The application requires a trained model in `models/absa_model/`. If you have not trained the model yet, complete Step 1 before running the app.

### Step 1: Train the model (Google Colab with a GPU)

Training on a CPU is slow, so Google Colab with a GPU is recommended.

1. Open Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run the following commands in separate notebook cells.

Clone the repository:

```python
!git clone https://github.com/spiyush09/review_Aspect_Analyzer.git
%cd review_Aspect_Analyzer
```

Install the requirements:

```python
!pip install -q -r requirements.txt
```

Install the spaCy English model:

```python
!python -m spacy download en_core_web_sm
```

Check GPU availability:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

The output should be `True` when the GPU runtime is configured correctly.

Train the model:

```python
!python -m src.train
```

Training downloads the dataset if necessary, fine-tunes BERT, and saves the trained model in `models/absa_model/` and evaluation results in `results/`.

**Download the trained model and results** using the following notebook cell:

```python
!zip -r output.zip models/ results/

from google.colab import files
files.download("output.zip")
```

Extract `output.zip` inside your local project directory. Confirm that `models/absa_model/` contains `config.json` and the trained model weights.

### Step 2: Run the app on Linux

Open a terminal in the project directory:

```bash
cd ~/Projects/review_Aspect_Analyzer
source .venv/bin/activate
```

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

Open the local URL printed in the terminal, usually:

http://localhost:8501

### Step 3: Run the app on Windows

Open PowerShell in the project directory and activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the application:

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL printed in the terminal, usually:

http://localhost:8501

### Using the application

1. Type or paste a customer review into the text box, or choose a ready-made example.
2. Click **Analyze review**.
3. View the extracted aspects, predicted sentiments, confidence scores, and corresponding sentences.

If the app reports that no trained model was found, check that the model files are present in `models/absa_model/`.

To try another model (for example, `roberta-base`), change `MODEL_NAME` in `config.py` and train again.

## 7. Results

Saved in the `results/` folder after training:

- `metrics.json`: validation and test accuracy / macro-F1, and the training settings
- `classification_report.txt`: precision, recall and F1 for each class on the test set
- `confusion_matrix.png`: confusion matrix on the test set

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| bert-base-uncased | 0.7657 | 0.6761 | 0.7726 | 0.7089 |

Training settings: 4 epochs, learning rate 2e-5, batch size 16, seed 42.

Accuracy is higher than macro-F1 because the data has many more positive examples than neutral ones, and the model finds the neutral class hardest.

## 8. Limitations

- Aspect extraction is rule based, so it can miss aspects written as verbs or adjectives ("it overheats") and sometimes picks nouns that are not real aspects.
- The model is trained on restaurant reviews only, so it may not work as well on other products.
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
