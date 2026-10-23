# Phishing Email Detection Using Machine Learning

A beginner-friendly machine-learning project that classifies email text as **phishing** or **legitimate**.

## Features
- Streamlit web interface
- TF-IDF text features with unigrams and bigrams
- Logistic Regression classifier
- Prediction probabilities and a simple test-set evaluation
- Option to upload your own CSV dataset

## Project structure

```text
phishing_email_detection_project/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── phishing_emails.csv
```

## Requirements
- Python 3.9 or newer
- Internet access for installing the Python packages

## Run locally

1. Extract the ZIP file.
2. Open a terminal or command prompt in the project folder.
3. Create and activate a virtual environment (recommended):

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local address shown in the terminal (usually `http://localhost:8501`).

## Dataset format

The CSV must contain these columns:

| email_text | label |
|---|---|
| Your account is locked. Click this link to verify your password. | phishing |
| The team meeting is scheduled for 10 AM tomorrow. | legitimate |

Accepted labels:
- `phishing`, `spam`, or `1`
- `legitimate`, `ham`, `safe`, or `0`

To use your own dataset, choose **Browse files** in the sidebar and upload a CSV with the required columns.

## How it works
1. Email text is cleaned and checked.
2. TF-IDF converts words and word pairs into numeric features.
3. Logistic Regression learns patterns from labeled examples.
4. The app predicts a class for the supplied message and shows model probabilities.

## Important limitations
The included dataset is small and created for demonstration, not production security. Its accuracy score is not a reliable estimate of real-world performance. For a meaningful academic evaluation, use a larger, well-labeled dataset, remove duplicates, avoid data leakage, and evaluate precision, recall, F1-score, and a confusion matrix. Email text alone cannot prove an email is safe; inspect sender domains, URLs, attachments, and authentication signals as well.

## Suggested project report sections
- Abstract
- Introduction and problem statement
- Objectives
- Literature review
- Dataset description and preprocessing
- TF-IDF feature extraction
- Logistic Regression methodology
- Results and evaluation
- Limitations and future scope
- Conclusion
