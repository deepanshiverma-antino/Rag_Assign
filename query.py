import faiss
import pickle
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "vectorstore"

def search(query, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(f"{VECTOR_DIR}/faiss.index")
    with open(f"{VECTOR_DIR}/metadata.pkl", "rb") as f:
        texts, metadata = pickle.load(f)

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(texts[i])

    return results

if __name__ == "__main__":
    q = input("Ask a question: ")
    passages = search(q)

    for i, p in enumerate(passages, 1):
        print(f"\n--- Result {i} ---\n{p}")
