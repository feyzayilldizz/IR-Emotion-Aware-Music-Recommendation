import math
from collections import Counter, defaultdict
from emotion import cosine_similariy
import numpy as np
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.N
        self.doc_freq = defaultdict(int)
        self.doc_len= []
        self.term_freq = []

        self._build_index()

    def _build_index(self):
        for doc in self.corpus:
            self.doc_len.append(len(doc))
            tf = Counter(doc)
            self.term_freq.append(tf)

            for term in tf:
                self.doc_freq[term] += 1

    def idf(self, term):
        df = self.doc_freq.get(term,0)
        if df == 0:
            return 0
        return math.log(1 + (self.N - df +0.5) / (df +0.5))
    
    def score(self, query_tokens, doc_id):
        score = 0
        tf_doc = self.term_freq[doc_id]
        doc_len = self.doc_len[doc_id]

        for term in query_tokens:
            if term not in tf_doc:
                continue

            f = tf_doc[term]
            idf = self.idf(term)

            denom = f + self.k1 * ( 1-self.b +self.b*doc_len / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        
        return score
    
    def rank(self, query_tokens, top_k =10):
        scores = []
        for doc_id in range(self.N):
            s = self.score(query_tokens, doc_id)
            if s > 0 :
                scores.append((doc_id, s))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class EmotionBM25:
    def __init__(self, bm25, emotion_vectors, alpha=0.3):
        self.bm25 = bm25
        self.emotion_vectors = emotion_vectors
        self.alpha = alpha

    def rank(self, query_tokens, query_emotion_vector,top_k =10):
        scores = []

        for doc_id in range(self.bm25.N):
            bm25_score = self.bm25.score(query_tokens, doc_id)
            emo_score = cosine_similariy(query_emotion_vector, self.emotion_vectors[doc_id])

            final_score = (
                self.alpha*bm25_score + (1-self.alpha)*emo_score
            )

            if final_score > 0:
                scores.append((doc_id, final_score))

        scores.sort(key= lambda x: x[1], reverse=True)
        return scores[:top_k]
    
class SemanticRetrieval:
    def __init__(self, embeddings, doc_ids=None):
        self.embeddings = embeddings
        self.doc_ids = doc_ids if doc_ids is not None else np.arange(embeddings.shape[0])
        self.norms = np.linalg.norm(embeddings, axis=1)

    def rank(self, query_embedding,top_k =10):
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        # Cosine similarity
        sims = np.dot(self.embeddings, query_embedding) / (self.norms * query_norm)

        top_indices = sims.argsort()[::-1][:top_k]

        return [(self.doc_ids[i], sims[i]) for i in top_indices]


class SemanticEmotionRetrieval:
    #  semantic similarity (Sbert) with emotion alignment
    
    def __init__(self, embeddings, emotion_vectors, alpha =0.5, doc_ids=None):
        self.embeddings = embeddings
        self.emotion_vectors = emotion_vectors
        self.alpha= alpha
        self.doc_ids = doc_ids if doc_ids is not None else np.arange(embeddings.shape[0])
        self.norms = np.linalg.norm(embeddings, axis=1)

    def rank(self, query_embedding, query_emotion_vector, top_k =10):
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        # Cosine similarity for semantic
        semantic_sims = np.dot(self.embeddings, query_embedding) / (self.norms * query_norm + 1e-8)

        # Cosine similarity for emotion
        emo_sims = np.array([
            cosine_similariy(query_emotion_vector, ev)
            for ev in self.emotion_vectors
        ])

        # Combined score
        final_scores = self.alpha * semantic_sims + (1 - self.alpha) * emo_sims

        top_indices = final_scores.argsort()[::-1][:top_k]

        return [(self.doc_ids[i], final_scores[i]) for i in top_indices]