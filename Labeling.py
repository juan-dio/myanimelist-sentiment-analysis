import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER resource
nltk.download('vader_lexicon')

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Labeling", page_icon=":label:")

# Fungsi untuk memuat dataset
@st.cache_data(persist=True)
def load_data(filepath):
  return pd.read_csv(filepath)

# Inisialisasi VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Fungsi untuk menghitung polarity score
def polarity_score(text):
  scores = sia.polarity_scores(str(text))  # Convert to string in case of missing values
  return scores['compound']

# Fungsi untuk memberikan label sentiment
def label_sentiment(score):
  if score >= 0.05:
    return 'positive'
  elif score <= -0.05:
    return 'negative'
  else:
    return 'neutral'

# Judul halaman
st.write("#### Juan Axl Ronaldio Zaka Putra (220411100066)")
st.title('MyAnimeList Review Sentiment Analysis')
st.write('## Labeling')

# Memuat dataset
df = load_data('./datasets/anime_reviews.csv')

# Menampilkan dataset
st.subheader('Dataset')
st.dataframe(df)

# Jalankan fungsi untuk menambahkan kolom 'score' pada dataset
df['score'] = df['review'].apply(polarity_score)

# Menampilkan dataset dengan kolom 'score'
st.subheader('Dataset with Polarity Score')
st.dataframe(df)

# Jalankan fungsi untuk menambahkan kolom 'sentiment' pada dataset
df['sentiment'] = df['score'].apply(label_sentiment)

# Menampilkan dataset dengan kolom 'sentiment'
st.subheader('Dataset with Sentiment Label')
st.dataframe(df)
