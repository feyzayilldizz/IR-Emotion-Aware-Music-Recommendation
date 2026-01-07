# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_text
from emotion import load_nrc_lexicon, emotion_vector, emotion_trigger_words
from retrieval import SemanticEmotionRetrieval
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def download_nltk():
    import nltk
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)
    return True

download_nltk()


# Load precomputed data
@st.cache_data
def load_data():
    df = pd.read_pickle("data/df_tokens.pkl")
    return df

@st.cache_resource
def load_models_and_vectors():
    # Load SBERT model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load precomputed embeddings and emotion vectors
    lyrics_embeddings = np.load("data/lyrics_embeddings.npy")
    emotion_vectors = np.load("data/emotion_vectors.npy",allow_pickle=True)

    # Load NRC Emotion Lexicon
    lexicon = load_nrc_lexicon("data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

    # Semantic + Emotion 
    semantic_emotion_model = SemanticEmotionRetrieval(
        embeddings=lyrics_embeddings,
        emotion_vectors=emotion_vectors,
        alpha=0.5
    )

    return sbert_model, lexicon, semantic_emotion_model

df = load_data()
sbert_model, lexicon, semantic_emotion_model = load_models_and_vectors()


st.title("🎵 Emotion-Aware Music Search")
st.write("A Song recommendation system based on open-text query")

query = st.text_input("How do you feel today?")
alpha = st.slider("Semantic vs Emotion weight (alpha)", 0.0, 1.0, 0.5, 0.05)
top_k = st.number_input("Number of results", min_value=1, max_value=20, value=5)


# retrieval

if query:
    tokens = preprocess_text(query)
    q_emo = emotion_vector(tokens, lexicon)
    q_emb = sbert_model.encode([query], convert_to_numpy=True)[0]

    semantic_emotion_model.alpha = alpha

    results = semantic_emotion_model.rank(q_emb, q_emo, top_k=top_k)

    if results:
        st.subheader("Top Songs:")
        table = []
        for rank, (doc_id, score) in enumerate(results, start=1):
            table.append({
                "Rank": rank,
                "Track": df.loc[doc_id, "track_name"],
                "Artist": df.loc[doc_id, "track_artist"],
                "Score": f"{score:.3f}"
            })
        st.table(pd.DataFrame(table))


        # Emotion plot
        st.subheader("Emotion Comparison")
        song_emotions = [df.loc[doc_id, "emotion_vec"] for doc_id, _ in results[:5]]
        song_names = [df.loc[doc_id, "track_name"] for doc_id, _ in results[:5]]

        EMOTIONS = ["anger", "fear", "anticipation", "trust",
                    "surprise", "sadness", "joy", "disgust"]

        fig, ax = plt.subplots(figsize=(10,5))
        query_vec = np.array(q_emo)
        for vec, name in zip(song_emotions, song_names):
            ax.plot(EMOTIONS, vec, marker='o', label=name)
        ax.plot(EMOTIONS, query_vec, marker='x', color='black', linestyle='--', label="Query")
        ax.set_ylabel("Emotion intensity")
        ax.set_title("Query vs Top Songs Emotion Profiles")
        ax.legend()
        st.pyplot(fig)

        # emotion-trigger words
        st.subheader("Top Emotion-Trigger Words in Query")
        top_words = emotion_trigger_words(tokens, lexicon, top_k=5)
        st.write([w for w, c in top_words])
    else:
        st.write("No results found.")
