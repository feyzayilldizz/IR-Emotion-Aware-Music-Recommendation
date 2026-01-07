# main.py
import pandas as pd
import numpy as np
import re
from preprocessing import preprocess_text
from emotion import load_nrc_lexicon, emotion_vector, emotion_trigger_words
from retrieval import BM25, EmotionBM25, SemanticRetrieval, SemanticEmotionRetrieval
from sentence_transformers import SentenceTransformer


# Load and filter dataset

df = pd.read_csv("data/spotify_songs.csv")
df = df.dropna(subset=["lyrics"])
df = df[df["language"] == "en"].reset_index(drop=True)

# Remove remixes 

def clean_track_name(name):
    # Remove remix info from track names
    name = re.sub(r"\s*-\s*.*Remix", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*\(.*Remix.*\)", "", name, flags=re.IGNORECASE)
    return name.strip()

df["clean_name"] = df["track_name"].apply(clean_track_name)
df["unique_id"] = df["clean_name"] + " - " + df["track_artist"]

df = df.drop_duplicates(subset="unique_id").reset_index(drop=True)

#  Preprocess lyrics

df["tokens"] = df["lyrics"].apply(preprocess_text)
df = df[df["tokens"].str.len() > 0].reset_index(drop=True)

#  Load emotion lexicon & compute vectors

lexicon = load_nrc_lexicon("data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
df["emotion_vec"] = df["tokens"].apply(lambda t: emotion_vector(t, lexicon))


# BM25 baseline
corpus = df["tokens"].tolist()
bm25 = BM25(corpus)

# Emotion-BM25
emotion_vectors = df["emotion_vec"].tolist()
emotion_bm25 = EmotionBM25(
    bm25=bm25,
    emotion_vectors=emotion_vectors,
    alpha=0.3
)


# Semantic 
print("Computing sentence embeddings for lyrics...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
lyrics_corpus = df["lyrics"].tolist()
lyrics_embeddings = sbert_model.encode(lyrics_corpus, convert_to_numpy=True, show_progress_bar=True)

semantic_retrieval = SemanticRetrieval(lyrics_embeddings)

# Semantic + Emotion 
semantic_emotion = SemanticEmotionRetrieval(
    embeddings=lyrics_embeddings,
    emotion_vectors=df["emotion_vec"].tolist(),
    alpha=0.5
)


# Test queries
queries_df = pd.read_excel("data/queries.xlsx")
queries = queries_df["query_text"].tolist()


# Run retrieval for annotation -will be runned ony once
def save_qrels(top_k=5, path="data/qrels.xlsx"):
    rows = []

    for qid, query in enumerate(queries):
        tokens = preprocess_text(query)
        q_emo = emotion_vector(tokens, lexicon)
        q_emb = sbert_model.encode([query], convert_to_numpy=True)[0]

       
        def deduplicate(ranked_docs):
            seen = set()
            filtered = []
            for doc_id, score in ranked_docs:
                uid = df.loc[doc_id, "unique_id"]
                if uid not in seen:
                    seen.add(uid)
                    filtered.append((doc_id, score))
                if len(filtered) >= top_k:
                    break
            return filtered

        for model_name, results in {
            "BM25": bm25.rank(tokens, top_k),
            "EmotionBM25": emotion_bm25.rank(tokens, q_emo, top_k),
            "Semantic": semantic_retrieval.rank(q_emb, top_k),
            "SemanticEmotion": semantic_emotion.rank(q_emb, q_emo, top_k)
        }.items():
            results = deduplicate(results)
            for rank, (doc_id, score) in enumerate(results, start=1):
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "model": model_name,
                    "rank": rank,
                    "doc_id": doc_id,
                    "track_name": df.loc[doc_id, "track_name"],
                    "artist": df.loc[doc_id, "track_artist"],
                    "score": score,
                    "relevant": ""  # Filled manually
                })

    annotation_df = pd.DataFrame(rows)
    annotation_df.to_excel(path, index=False)


# save_qrels()

