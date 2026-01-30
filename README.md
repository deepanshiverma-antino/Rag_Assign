# RAG_Assign

Simple Retrieval-Augmented Generation (RAG) project used for an assignment.

Contents

- `extract.py` — extract text from `docs/` into a format suitable for embedding/indexing
- `ingest.py` — ingest documents and (optionally) build a FAISS index
- `query.py` — run queries against the index to retrieve relevant passages
- `docs/` — source text files used for experiments
- `faiss_index/` — local FAISS index files (ignored by Git)

Quick start

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate # macOS/Linux
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Ingest documents and build an index (adjust to your environment):
   ```bash
   python ingest.py
   ```
4. Run queries:
   ```bash
   python query.py
   ```

Notes

- The `faiss_index/` folder is intentionally ignored to avoid pushing large binary index files. If you need to share an index, consider using Git LFS or an external artifact store.
- Feel free to add a `README` section describing dataset sources, usage examples, or tests.

License
This project is licensed under the MIT License (see `LICENSE`).
