from pathlib import Path

import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "passages.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_data
def load_knowledge_base():

    df = pd.read_csv(DATA_PATH)

    df["text"] = df["text"].fillna("").astype(str)

    return df


@st.cache_resource
def load_model():

    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def build_passage_embeddings(_df, _model):

    passages = _df["text"].tolist()

    embeddings = _model.encode(
        passages,
        convert_to_numpy=True
    )

    return embeddings

def build_tfidf_index(df):
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_embeddings = vectorizer.fit_transform(
        df["text"].tolist()
    )

    return vectorizer, tfidf_embeddings

def extract_relevant_sentences(question, passages, max_sentences=6):
    """
    Extract sentences from retrieved passages that are most relevant
    to the user's question.
    """

    question_embedding = None

    # Combine all retrieved passages into individual sentences
    candidates = []

    for passage in passages:
        sentences = re.split(r'(?<=[.!?])\s+', passage["text"])

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                candidates.append({
                    "text": sentence,
                    "title": passage["title"],
                    "passage_id": passage["passage_id"]
                })

    if not candidates:
        return []

    # Use TF-IDF to compare the question with candidate sentences
    vectorizer = TfidfVectorizer(stop_words="english")

    texts = [question] + [candidate["text"] for candidate in candidates]

    tfidf_matrix = vectorizer.fit_transform(texts)

    question_vector = tfidf_matrix[0]
    sentence_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(
        question_vector,
        sentence_vectors
    )[0]

    ranked_indices = np.argsort(scores)[::-1]

    selected = []

    for index in ranked_indices:

        candidate = candidates[index]

        selected.append({
            "text": candidate["text"],
            "title": candidate["title"],
            "passage_id": candidate["passage_id"],
            "score": float(scores[index])
        })

        if len(selected) >= max_sentences:
            break

    return selected

def retrieve(
    question,
    df,
    model,
    passage_embeddings,
    tfidf_vectorizer,
    tfidf_embeddings,
    k=3
):
    # --------------------------------------------
    # Dense similarity
    # --------------------------------------------

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    dense_scores = cosine_similarity(
        question_embedding,
        passage_embeddings
    )[0]

    # --------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------

    question_tfidf = tfidf_vectorizer.transform(
        [question]
    )

    tfidf_scores = cosine_similarity(
        question_tfidf,
        tfidf_embeddings
    )[0]

    # --------------------------------------------
    # Hybrid score
    # --------------------------------------------

    hybrid_scores = (
        0.7 * dense_scores
        + 0.3 * tfidf_scores
    )

    top_indices = np.argsort(
        hybrid_scores
    )[-k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "passage_id": df.iloc[index]["id"],
            "title": df.iloc[index]["title"],
            "url": df.iloc[index]["url"],
            "text": df.iloc[index]["text"],
            "score": float(hybrid_scores[index])
        })

    return results