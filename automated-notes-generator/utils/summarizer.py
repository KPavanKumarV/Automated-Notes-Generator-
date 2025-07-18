# utils/summarizer.py

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_chunk=1000):
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return "\n".join(summaries)

def extract_keywords(text, top_k=10):
    tfidf = TfidfVectorizer(stop_words='english', max_features=top_k)
    X = tfidf.fit_transform([text])
    return tfidf.get_feature_names_out()
