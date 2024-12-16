import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configure Streamlit page
st.set_page_config(page_title="Modeling", page_icon=":bar_chart:")

# Function to load dataset
@st.cache_data(persist=True)
def load_data(filepath):
  return pd.read_csv(filepath)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def preprocess_text(text):
  text = text.lower()  # Lowercase
  text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
  text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])  # Remove stopwords
  text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])  # Lemmatization
  return text

# Page title and description
st.write("#### Juan Axl Ronaldio Zaka Putra (220411100066)")
st.title('MyAnimeList Review Sentiment Analysis')
st.write('## Modeling')

# Load dataset
df = load_data('./datasets/top_anime_reviews_labeled.csv')
data = df.copy()

# Display dataset
st.subheader('Dataset')
st.write(data)

# Preprocess text data
data['review'] = data['review'].apply(preprocess_text)

# Encode sentiment labels
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else (0 if x == 'neutral' else -1))

# Display dataset after preprocessing
st.subheader('Dataset after Preprocessing')
st.write(data)

# Split data into training and testing sets
X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with TfidfVectorizer and LogisticRegression
model = Pipeline([
  ('tfidf', TfidfVectorizer()),
  ('lr', LogisticRegression(class_weight='balanced'))
])

# Train the model
model.fit(X_train, y_train)

# Model evaluation
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(3, 3))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
st.write("### Classification Report")
st.write(pd.DataFrame(report).transpose())
