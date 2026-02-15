# src/retrieval/search.py
from .resume_parser import load_resumes
from .embeddings import EmbeddingModel
from .faiss_index import ResumeIndex

# Initialize components
embedder = EmbeddingModel()
EMBEDDING_DIMENSION = embedder.get_embedding_dimension()
vector_db = ResumeIndex(dimension=EMBEDDING_DIMENSION)

# FIX: Try to load existing index immediately
index_loaded = vector_db.load_index()
if index_loaded and vector_db.dimension != EMBEDDING_DIMENSION:
    raise RuntimeError(
        "Loaded index dimension does not match current embedding model. "
        f"Model dim={EMBEDDING_DIMENSION}, index dim={vector_db.dimension}. "
        "Rebuild or replace vector_store/index.faiss."
    )

def ingest_resumes(resume_folder_path):
    """
    Full pipeline: Parse -> Embed -> Index -> Save
    """
    raw_data = load_resumes(resume_folder_path)
    if not raw_data:
        print("No resumes found to ingest.")
        return

    # Prevent duplicate indexing across reruns and inside the same batch.
    existing_ids = {
        data.get("id") for data in vector_db.metadata_map.values() if data.get("id")
    }
    new_data = []
    seen_in_batch = set()
    for item in raw_data:
        item_id = item.get("id")
        if not item_id:
            continue
        if item_id in existing_ids or item_id in seen_in_batch:
            continue
        new_data.append(item)
        seen_in_batch.add(item_id)

    if not new_data:
        print("No new resumes to add.")
        return

    texts = [item["text"] for item in new_data]
    
    print("Generating embeddings...")
    vectors = embedder.get_embeddings(texts)

    vector_db.add_resumes(vectors, new_data)
    
    # FIX: Save index immediately after adding
    vector_db.save_index()

def query_resumes(query_text, top_k=3):
    query_vec = embedder.get_query_embedding(query_text)
    results = vector_db.search(query_vec, k=top_k)
    return results

if __name__ == "__main__":
    TEST_RESUME_FOLDER = "Resume" 
    
    # Run ingestion (this will now save to disk)
    ingest_resumes(TEST_RESUME_FOLDER)
    
    user_query = "machine learning engineer with python"
    matches = query_resumes(user_query)
    
    print(f"\nQuery: {user_query}")
    print("-" * 30)
    for match in matches:
        print(f"File ID: {match['resume'].get('id')}") # Now using Hash ID
        print(f"Filename: {match['resume'].get('filename')}")
        print(f"Score: {match['score']:.4f}") # Now Cosine Similarity (Higher is better)
        print("-" * 10)
