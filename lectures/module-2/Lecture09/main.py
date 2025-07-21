from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import heapq
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

nltk.download("punkt")
nltk.download("stopwords")

app = FastAPI(title="Text Summarization API with ROUGE Score")

# Temizleme fonksiyonu
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    return text

# Özetleme fonksiyonu
def summarize_text(text: str, num_sentences: int = 3) -> str:
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(cleaned_text)
    stop_words = stopwords.words("english")

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(sentences)

    sentence_scores = {sentences[i]: X[i].toarray().sum() for i in range(len(sentences))}
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    return ' '.join(summary_sentences)

# ROUGE skorunu hesaplayan fonksiyon
def compute_rouge(reference: str, summary: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {k: v.fmeasure for k, v in scores.items()}

# İstek yapısı
class TextRequest(BaseModel):
    text: str
    reference: Optional[str] = None
    num_sentences: Optional[int] = 3

@app.post("/summarize/")
def summarize(req: TextRequest):
    summary = summarize_text(req.text, req.num_sentences)
    
    result = {"summary": summary}

    # Eğer referans varsa ROUGE hesapla
    if req.reference:
        rouge_scores = compute_rouge(req.reference, summary)
        result["rouge"] = rouge_scores

    return result
