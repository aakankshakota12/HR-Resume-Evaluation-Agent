# src/retrieval/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        """
        Converts text to normalized vectors (required for Cosine Similarity).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Safe normalization: avoid division-by-zero for degenerate vectors.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        
        return embeddings.astype('float32')
    
    def get_query_embedding(self, query):
        """
        Converts query to normalized vector.
        """
        embedding = self.model.encode([query], convert_to_numpy=True)

        # Same zero-safe normalization for query embedding.
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = embedding / norms
        
        return embedding.astype('float32')
