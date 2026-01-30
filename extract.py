import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

def extract_answer(question, passages, top_n=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    q_emb = model.encode([question])

    scored = []
    for p in passages:
        sentences = sent_tokenize(p)
        sent_embs = model.encode(sentences)

        sims = cosine_similarity(q_emb, sent_embs)[0]
        for s, score in zip(sentences, sims):
            scored.append((score, s))

    scored.sort(reverse=True)
    return [s for _, s in scored[:top_n]]
