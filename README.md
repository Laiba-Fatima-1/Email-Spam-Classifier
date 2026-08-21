# The Sorting Desk — Email Spam Classifier

**Task 1 · Arch Technologies Machine Learning Internship · Month 1**

A trained ML model (TF-IDF + Multinomial Naive Bayes) wrapped in a Streamlit web app.
Paste any email/message text and get an instant Spam / Safe verdict — stamped like a
piece of mail passing through a sorting office — with a confidence score and the
specific words that drove the decision.

**Live app:** https://email-spam-sorter.streamlit.app/
**Repository:** https://github.com/Laiba-Fatima-1/Email-Spam-Classifier

## Results
- Accuracy: 93.5%
- Precision: 95.1%
- Recall: 91.0%
- F1-score: 93.0%

Evaluated on a held-out test set of 38,770 emails the model never saw during training.

These numbers are lower than a smaller, easier dataset would produce — that's expected
and a good sign, not a regression. This dataset has 194K real, varied emails (spam
ranging from obvious to subtle, ham including everything from casual messages to dense
business correspondence), so it's a genuinely harder and more realistic test than a
small, uniform dataset would be.

## IMPORTANT — about the dataset file (do this before pushing to GitHub)
`spam_emails_kaggle.csv` is ~357 MB. GitHub blocks any file over 100 MB, so **do not
try to push this CSV to your repository** — the push will fail.

You don't need to: the trained model (`spam_model.joblib`) and vectorizer
(`vectorizer.joblib`) are both under 200 KB and contain everything the app needs to
run. The raw CSV is only needed if you want to retrain from scratch — keep it on your
own machine, just don't commit it to git.

If your `.gitignore` doesn't already exclude it, add this line to a `.gitignore` file
in the project folder:
```
spam_emails_kaggle.csv
```

## How to run it locally
1. Open a terminal in this folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py
   (or `python -m streamlit run app.py` if `streamlit` isn't on your PATH)
4. It opens automatically in your browser (usually http://localhost:8501)

## Files
- app.py             -> the web app (run this)
- train_model.py     -> the training script (already run once — the files below are its output)
- spam_model.joblib  -> trained model (loaded by app.py)
- vectorizer.joblib  -> trained TF-IDF vectorizer (loaded by app.py)
- metrics.json       -> saved evaluation metrics (shown in the app + sidebar)
- spam_emails_kaggle.csv -> the dataset used for training (DO NOT commit — see above)

## Retraining (optional)
If you want to retrain from scratch (needs spam_emails_kaggle.csv present locally):
   python train_model.py
This regenerates spam_model.joblib, vectorizer.joblib, and metrics.json.

## Approach
1. Clean each email: lowercase, strip URLs and punctuation, collapse whitespace.
2. Convert text to numeric features using TF-IDF (5,000 features, English stop words
   removed) — weights words by how distinctive they are to a message, not just how
   often they appear.
3. Train a Multinomial Naive Bayes classifier — the standard, interpretable algorithm
   for spam filtering.
4. Evaluate on a 20% held-out test split (38,770 emails) never seen during training.

## Dataset
"190K+ Spam | Ham Email Dataset for Classification" (Kaggle, by meruvulikith) —
193,850 labeled real emails (102,159 legitimate, 91,691 spam), sourced from the Enron
and related email corpora. Columns: `label` (Spam/Ham), `text` (raw email content).
