# src/retrieval/faiss_index.py
import faiss
import pickle
import os
import numpy as np

class ResumeIndex:
    def __init__(self, dimension=384):
        # 384 is the dimension size for 'all-MiniLM-L6-v2'
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean Distance
        self.metadata_map = {} # Maps FAISS ID (int) -> Resume Data (dict)

    def add_resumes(self, embeddings, resumes_data):
        """
        Adds vectors to FAISS and stores metadata mapping.
        """
        if len(embeddings) != len(resumes_data):
            raise ValueError("Number of embeddings must match number of resumes.")
        
        # Add to FAISS
        self.index.add(embeddings)
        
        # Store metadata (mapping current index size back to data)
        # Note: FAISS IDs are sequential integers starting from 0
        start_id = self.index.ntotal - len(embeddings)
        for i, data in enumerate(resumes_data):
            self.metadata_map[start_id + i] = data
        
        print(f"Added {len(embeddings)} documents to index. Total: {self.index.ntotal}")

    def search(self, query_vector, k=5):
        """
        Searches the index for the k nearest neighbors.
        Returns: List of (resume_data, score)
        """
        # D is distances, I is indices
        D, I = self.index.search(query_vector, k)
        
        results = []
        # I[0] contains the indices of the top matches
        for j, idx in enumerate(I[0]):
            if idx != -1: # -1 indicates no match found
                results.append({
                    "resume": self.metadata_map.get(idx, {}),
                    "score": float(D[0][j]) # Lower score = closer distance = better match
                })
        return results

    def save_index(self, folder_path="vector_store"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata_map, f)
        print("Index and metadata saved.")

    def load_index(self, folder_path="vector_store"):
        index_path = os.path.join(folder_path, "index.faiss")
        meta_path = os.path.join(folder_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata_map = pickle.load(f)
            print("Index loaded from disk.")
        else:
            print("No existing index found. Starting fresh.")