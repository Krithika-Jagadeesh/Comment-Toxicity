import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re

# Load artifacts
model = tf.keras.models.load_model("toxicity_model/lstm_model.keras")
tokenizer = pickle.load(open("toxicity_model/tokenizer.pkl", "rb"))
thresholds = np.load("toxicity_model/best_thresholds.npy")

label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
max_len = 200

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def predict_labels(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    probs = model.predict(pad)[0]
    preds = (probs > thresholds).astype(int)
    return dict(zip(label_names, preds)), dict(zip(label_names, probs))

st.title("🧠 Toxic Comment Classifier")
st.markdown("Enter a comment to check for toxicity across multiple categories.")

# Single comment input
text_input = st.text_area("💬 Enter your comment here:")
if st.button("Analyze"):
    labels, scores = predict_labels(text_input)
    st.subheader("🔍 Prediction")
    for label in label_names:
        st.write(f"**{label}**: {'✅ Toxic' if labels[label] else '❎ Non-toxic'} (score: {scores[label]:.2f})")

# Bulk upload
st.markdown("---")
st.subheader("📁 Bulk Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with a 'comment' column", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'comment' not in df.columns:
        st.error("CSV must contain a 'comment' column.")
    else:
        df['cleaned'] = df['comment'].astype(str).apply(clean_text)
        seqs = tokenizer.texts_to_sequences(df['cleaned'])
        padded = pad_sequences(seqs, maxlen=max_len, padding='post')
        probs = model.predict(padded)
        preds = (probs > thresholds).astype(int)
        for i, label in enumerate(label_names):
            df[label] = preds[:, i]
            df[f"{label}_score"] = probs[:, i]
        st.dataframe(df.head())

        st.download_button("Download Results", df.to_csv(index=False), "toxicity_predictions.csv")
