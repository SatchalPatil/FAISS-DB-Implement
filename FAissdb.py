import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


TEXT_DIR = "Path to directory"  # Directory containing your .txt files
MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"

model = SentenceTransformer(MODEL_NAME)

def read_documents(directory):
    """
    Read all .txt files from a directory and return a list of Document objects.
    Each Document's page_content is the full file text, and metadata contains the file name.
    """
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                doc = Document(page_content=text, metadata={"file_name": file_name})
                documents.append(doc)
    return documents

docs = read_documents(TEXT_DIR)
print(f"Loaded {len(docs)} document(s).")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"Split documents into {len(split_docs)} chunks.")

#list of text chunks and a metadata list.
#copy each chunk's text into its metadata under the key "text" so that queries can display it.
all_chunks = [doc.page_content for doc in split_docs]
all_metadata = []
for doc in split_docs:
    meta = doc.metadata.copy()  
    meta["text"] = doc.page_content  
    all_metadata.append(meta)

embeddings = model.encode(all_chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)

#Save the metadata
with open(METADATA_FILE, 'wb') as f:
    pickle.dump(all_metadata, f)

print(f"FAISS index created with {index.ntotal} vectors")
print(f"Index saved to {INDEX_FILE}")
print(f"Metadata saved to {METADATA_FILE}")
