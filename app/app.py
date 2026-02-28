import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# LOAD MODEL
# =========================
MODEL_DIR = "/content/drive/MyDrive/skripsi/model"

svm_sentimen = joblib.load(f"{MODEL_DIR}/svm_sentimen.pkl")
tfidf_sentimen = joblib.load(f"{MODEL_DIR}/tfidf_sentimen.pkl")

svm_aspek = joblib.load(f"{MODEL_DIR}/svm_aspek.pkl")
tfidf_aspek = joblib.load(f"{MODEL_DIR}/tfidf_aspek.pkl")

# =========================
# INIT STEMMER
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stem_text(text):
    return stemmer.stem(text)

# =========================
# STREAMLIT UI
# =========================
st.title("📊 Analisis Sentimen & Aspek Ulasan Aplikasi")

user_input = st.text_area("Masukkan ulasan pengguna:")

if st.button("Analisis"):
    if user_input.strip() == "":
        st.warning("Masukkan teks dulu")
    else:
        cleaned = clean_text(user_input)
        stemmed = stem_text(cleaned)

        # Sentimen
        vec_sent = tfidf_sentimen.transform([cleaned])
        sentimen = svm_sentimen.predict(vec_sent)[0]

        # Aspek
        vec_asp = tfidf_aspek.transform([stemmed])
        aspek = svm_aspek.predict(vec_asp)[0]

        st.subheader("🔎 Hasil Analisis")
        st.write(f"**Sentimen:** {sentimen}")
        st.write(f"**Aspek:** {aspek}")
