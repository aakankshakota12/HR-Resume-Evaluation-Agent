# src/retrieval/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the embedding model. 
        'all-MiniLM-L6-v2' is fast and good for generic retrieval.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        """
        Converts a list of text strings into a numpy array of embeddings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # FAISS expects float32
        return embeddings.astype('float32')
    
    def get_query_embedding(self, query):
        """
        Converts a single query string into a 2D numpy array (1, dimension).
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding.astype('float32')