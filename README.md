# Fake News Detection

A machine-learning project that classifies a news article as **Fake** or **Real** from its headline and article text. It includes a Jupyter notebook for data preparation, model training, and evaluation, plus a Streamlit app for interactive predictions.

> **Important:** This is a text classifier, not a fact-checking service. Its predictions and confidence scores do not establish whether a claim is true.

## Features

- Prepares the Fake and Real news CSV files for classification.
- Cleans article text and converts it to TF-IDF features.
- Compares Logistic Regression and Multinomial Naive Bayes, with hyperparameter tuning using stratified cross-validation.
- Evaluates the selected model on a held-out test set.
- Serves predictions through a Streamlit interface.
- Displays the predicted class and, when supported by the classifier, its estimated probability for that class.

## Project structure

```text
.
├── app.py                          # Streamlit prediction application
├── det.ipynb                       # Data preparation, training, and evaluation
├── requirements.txt                # Pinned Python dependencies
├── Data/
│   ├── Fake.csv                    # Fake news articles
│   └── True.csv                    # Real news articles
└── model_artifacts/
    ├── fake_news_model.pkl         # Trained classifier
    └── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
```

The notebook documents 23,502 fake articles and 21,417 real articles. Each dataset contains `Title`, `Text`, `Subject`, and `Date` columns. Training combines the title and article text; the Streamlit app asks for those same two fields.

## Requirements

- Python compatible with the pinned packages in `requirements.txt`
- pip
- Internet access on the first app launch if the NLTK English stopwords corpus is not already installed; the app downloads it automatically

The dependencies are pinned in `requirements.txt`. The file includes packages used by the application and notebook, including Streamlit, NLTK, pandas, scikit-learn, and Jupyter.

## Setup and run

Run these commands from the project root.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL when the server starts; open it in a browser. Enter both a headline and article body, then select **Analyze News**.

The app loads the model and fitted vectorizer from `model_artifacts/`. If those files are missing, run the notebook through its artifact-saving cell before starting the app.

## Train or reproduce the model

1. Install the dependencies above.
2. Confirm `Data/Fake.csv` and `Data/True.csv` are present.
3. Open `det.ipynb` in VS Code with its Jupyter extension, JupyterLab, or another notebook environment. `requirements.txt` includes the notebook kernel dependencies, but not a Jupyter Notebook or JupyterLab interface; install your preferred interface separately if needed.
4. Run the notebook cells in order. It loads and combines the CSV files, cleans text, creates a stratified 80/20 train/test split, fits a TF-IDF vectorizer, compares and tunes Logistic Regression and Multinomial Naive Bayes, and reports evaluation metrics.
5. Run the final artifact-saving cell. It writes the selected classifier to `model_artifacts/fake_news_model.pkl` and the fitted vectorizer to `model_artifacts/tfidf_vectorizer.pkl`.

Run the notebook with the project root as its working directory so its relative paths to `Data/` and `model_artifacts/` resolve correctly. The Streamlit app does not train a model; it uses the saved artifacts.

## How predictions work

The app combines the entered headline and article text, lowercases the text, removes URLs and HTML tags, retains alphabetic words, and removes English stopwords while keeping negation words (`no`, `nor`, `not`, and `never`). It then transforms the cleaned text with the saved TF-IDF vectorizer and passes the features to the saved classifier.

The classifier labels are:

| Label | App result |
|---|---|
| `0` | Fake News |
| `1` | Real News |

The displayed confidence, when available, is the classifier's estimated probability for its chosen class. It is not a guarantee that the classification is correct.

## Data and model files

The `Data/` directory is excluded by `.gitignore`, so the CSV datasets may need to be obtained and placed there separately. The trained artifacts are included under `model_artifacts/`; if they are not present in a checkout or you retrain the notebook, regenerate them as described above.

Only load joblib model files from a source you trust. Joblib artifacts use Python's pickle-based serialization and can execute code when loaded.
