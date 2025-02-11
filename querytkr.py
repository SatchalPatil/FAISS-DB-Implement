import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'  # Must match the model used for indexing

def load_resources():
    """Load FAISS index, metadata, and embedding model"""
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"FAISS index file {INDEX_FILE} not found")
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file {METADATA_FILE} not found")

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)
    
    # Load metadata
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)
    
    return index, metadata, model

def search(index, metadata, model, query, k=5):
    """Perform a search and return formatted results"""
    # Generate query embedding
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    
    # Search the index for k nearest neighbors
    distances, indices = index.search(query_embedding, k)
    
    # Prepare results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        meta = metadata[idx]
        file_name = meta.get('file_name', 'unknown')
        # Safely get the text; if not present, use a fallback message.
        text_val = meta.get('text')
        if text_val is not None:
            text_snippet = text_val[:2000] + '...'
        else:
            text_snippet = "[Text not stored in metadata]"
        result = {
            "rank": i + 1,
            "file_name": file_name,
            "text_snippet": text_snippet,
            "distance": float(distance)
        }
        results.append(result)
    
    return results

def main():
    try:
        # Load required resources
        index, metadata, model = load_resources()
        print(f"Successfully loaded index with {len(metadata)} documents")
        print("Model ready for encoding queries\n")
        
        # Interactive search loop
        while True:
            query = input("Enter your search query (or 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                print("Please enter a valid query\n")
                continue
            
            # Perform search
            results = search(index, metadata, model, query)
            
            # Display results
            print(f"\nResults for: '{query}'")
            for result in results:
                print(f"{result['rank']}. File: {result['file_name']}")
                print(f"   Text: {result['text_snippet']}")
                print(f"   Distance: {result['distance']:.4f}\n")
            
            print("-" * 50 + "\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
