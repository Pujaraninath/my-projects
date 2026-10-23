import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Phishing Email Detection", page_icon="🛡️", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "data" / "phishing_emails.csv"

st.title("🛡️ Phishing Email Detection Using Machine Learning")
st.write(
    "Enter an email message to estimate whether it looks like phishing or legitimate. "
    "This educational demo uses TF-IDF features and Logistic Regression."
)
st.warning(
    "Demo limitation: the included dataset is small and illustrative. Predictions are not a guarantee "
    "that an email is safe. Never open suspicious links or share passwords based only on this result."
)

@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATASET)

def prepare_dataset(df):
    required = {"email_text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain columns named 'email_text' and 'label'.")
    clean = df[["email_text", "label"]].dropna().copy()
    clean["email_text"] = clean["email_text"].astype(str).str.strip()
    clean["label"] = clean["label"].astype(str).str.strip().str.lower()
    label_map = {
        "phishing": "phishing", "spam": "phishing", "1": "phishing",
        "legitimate": "legitimate", "safe": "legitimate", "ham": "legitimate", "0": "legitimate"
    }
    clean["label"] = clean["label"].map(label_map)
    clean = clean.dropna()
    clean = clean[clean["email_text"].str.len() > 0]
    if clean["label"].nunique() != 2:
        raise ValueError("Dataset must include both phishing and legitimate examples.")
    if clean["label"].value_counts().min() < 2:
        raise ValueError("Please provide at least two examples for each class.")
    return clean

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["email_text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english",
                                  max_features=10000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, y_test, predictions

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload a CSV dataset (optional)", type=["csv"])
    st.caption("Required columns: email_text, label")
    st.caption("Labels accepted: phishing / legitimate (also spam / ham, 1 / 0).")

try:
    raw_df = pd.read_csv(uploaded_file) if uploaded_file is not None else load_default_data()
    df = prepare_dataset(raw_df)
    model, accuracy, y_test, predictions = train_model(df)
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Training dataset rows", f"{len(df):,}")
col2.metric("Phishing examples", f"{(df['label'] == 'phishing').sum():,}")
col3.metric("Test split accuracy", f"{accuracy:.1%}")
st.caption("The test accuracy is based on a small holdout split and should not be treated as real-world performance.")

st.subheader("Analyze an email")
email_text = st.text_area(
    "Paste the email body or message text",
    height=180,
    placeholder="Example: Your account will be suspended. Click here to verify your password..."
)
if st.button("Detect phishing", type="primary"):
    if not email_text.strip():
        st.info("Please paste an email message first.")
    else:
        predicted_label = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        class_names = list(model.named_steps["classifier"].classes_)
        prob_map = dict(zip(class_names, probabilities))
        confidence = prob_map[predicted_label]

        if predicted_label == "phishing":
            st.error(f"Prediction: LIKELY PHISHING  ·  Model confidence: {confidence:.1%}")
        else:
            st.success(f"Prediction: LIKELY LEGITIMATE  ·  Model confidence: {confidence:.1%}")

        st.progress(float(confidence), text=f"Confidence for predicted class: {confidence:.1%}")
        st.write("**Class probabilities**")
        st.write({
            "Phishing": f"{prob_map.get('phishing', 0):.1%}",
            "Legitimate": f"{prob_map.get('legitimate', 0):.1%}"
        })
        st.info("Check the sender address, domain spelling, unexpected attachments, urgency, and requests for credentials.")

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Evaluation details"):
    st.write(f"Holdout accuracy: **{accuracy:.1%}**")
    st.text(classification_report(y_test, predictions, zero_division=0))
    cm = confusion_matrix(y_test, predictions, labels=["legitimate", "phishing"])
    st.write("Confusion matrix (rows = actual, columns = predicted; order: legitimate, phishing)")
    st.dataframe(pd.DataFrame(cm,
        index=["Actual legitimate", "Actual phishing"],
        columns=["Predicted legitimate", "Predicted phishing"]))

st.markdown("---")
st.caption("Educational project · TF-IDF + Logistic Regression · Do not use as the only security control.")
