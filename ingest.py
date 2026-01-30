import os
import pickle
import faiss
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

UPLOAD_DIR = "data/uploads"
VECTOR_DIR = "vectorstore"

def load_documents():
    docs = []
    for file in Path(UPLOAD_DIR).glob("*"):
        if file.suffix == ".txt":
            docs.extend(TextLoader(str(file), encoding="utf-8").load())
        elif file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix == ".docx":
            docs.extend(Docx2txtLoader(str(file)).load())
    return docs

def ingest_documents():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    texts = [c.page_content for c in chunks]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, f"{VECTOR_DIR}/faiss.index")

    with open(f"{VECTOR_DIR}/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("✅ Ingestion complete")

if __name__ == "__main__":
    ingest_documents()
