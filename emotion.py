from collections import defaultdict, Counter
import numpy as np

EMOTIONS = [
    "anger", "fear", "anticipation", "trust",
    "surprise", "sadness", "joy", "disgust"
]

def load_nrc_lexicon(path: str):
    # dict[word] -> set(emotions)

    lexicon = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, value=line.strip().split("\t")
            if int(value) ==1:
                lexicon[word].add(emotion)

    return lexicon


def emotion_vector(tokens,lexicon):
    # Compute normalized emotion vector 

    vec = np.zeros(len(EMOTIONS))

    for token in tokens:
        if token in lexicon:
            for emotion in lexicon[token]:
                if emotion in EMOTIONS:
                    idx = EMOTIONS.index(emotion)
                    vec[idx] += 1

    if vec.sum()> 0:
        vec = vec/vec.sum()

    return vec

def cosine_similariy(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

def emotion_trigger_words(tokens, lexicon, top_k=10):
    # top emotion-triggering words in a document
    
    counts = Counter()

    for w in tokens:
        if w in lexicon:
            counts[w] += 1

    return counts.most_common(top_k)