# src/retrieval/faiss_index.py
import faiss
import pickle
import os
import numpy as np

class ResumeIndex:
    def __init__(self, dimension=384):
        self.dimension = dimension
        # FIX: Use Inner Product (IP) instead of L2. 
        # IP + Normalized Vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension) 
        self.metadata_map = {} 

    def add_resumes(self, embeddings, resumes_data):
        if len(embeddings) != len(resumes_data):
            raise ValueError("Number of embeddings must match number of resumes.")
        
        self.index.add(embeddings)
        
        start_id = self.index.ntotal - len(embeddings)
        for i, data in enumerate(resumes_data):
            self.metadata_map[start_id + i] = data
        
        print(f"Added {len(embeddings)} documents to index. Total: {self.index.ntotal}")

    def search(self, query_vector, k=5):
        # D is score (Cosine Similarity), I is indices
        D, I = self.index.search(query_vector, k)
        
        results = []
        for j, idx in enumerate(I[0]):
            if idx != -1: 
                results.append({
                    "resume": self.metadata_map.get(idx, {}),
                    "score": float(D[0][j]) # Higher score = Better match (Max 1.0)
                })
        return results

    def save_index(self, folder_path="vector_store"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata_map, f)
        print("Index and metadata saved to disk.")

    def load_index(self, folder_path="vector_store"):
        index_path = os.path.join(folder_path, "index.faiss")
        meta_path = os.path.join(folder_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata_map = pickle.load(f)
            print("Index loaded from disk.")
            return True
        else:
            print("No existing index found on disk.")
            return False