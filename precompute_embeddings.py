import pandas as pd
import numpy as np
from preprocessing import preprocess_text
from emotion import load_nrc_lexicon, emotion_vector
from sentence_transformers import SentenceTransformer

print("Loading dataset")
df = pd.read_csv("data/spotify_songs.csv")
df = df.dropna(subset=["lyrics"])
df = df[df["language"] == "en"].reset_index(drop=True)

print("Preprocessing tokens")
df["tokens"] = df["lyrics"].apply(preprocess_text)
df = df[df["tokens"].str.len() > 0].reset_index(drop=True)

print("Loading emotion lexicon and computing emotion vectors")
lexicon = load_nrc_lexicon("data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
df["emotion_vec"] = [emotion_vector(t, lexicon) for t in df["tokens"]]
emotion_vectors = np.array(df["emotion_vec"])

print("Loading SBERT model")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Computing embeddings")
lyrics_embeddings = model.encode(
    df["lyrics"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Saving embeddings and emotion vectors")
np.save("data/lyrics_embeddings.npy", lyrics_embeddings)
np.save("data/emotion_vectors.npy", emotion_vectors)
df.to_pickle("data/df_tokens.pkl")

print("Precomputed embeddings saved.")
