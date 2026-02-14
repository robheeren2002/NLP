# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code from class to create
# a chroma DB. ChatGPT 5.2 was used to help make this code
# work with larger json files.
# All chats can be found at:
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# ============================================================
import shutil
import os
import json
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------
# 1) Load docs file from JSON
# ---------------------------
def load_from_local(directory, filename):
    full_path = os.path.join(directory, filename)

    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}!")
        return []

    with open(full_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    # Rebuild the LangChain Document objects
    reconstructed_docs = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in loaded_data
    ]

    print(f"Loaded {len(reconstructed_docs)} documents from {full_path}")
    return reconstructed_docs

#adjust
docs_web_map = load_from_local(
    directory=r"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
              r"Processing\datasets\langchain_rag\backup",
    filename="docs_web_map_polestar_us.json"
)

if not docs_web_map:
    raise SystemExit("No documents loaded. Exiting.")


# ---------------------------
# 2) Define text splitters
# ---------------------------
splitter_pdf_papers = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024,
    chunk_overlap=128,
    separators=["\n\n", "\n", " ", ""]
)

splitter_web = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024,
    chunk_overlap=128,
    separators=["\n\n", "\n", " ", ""]
)

# Apply text splitter
chunks_web_map = splitter_web.split_documents(docs_web_map)
print(f"Split into {len(chunks_web_map)} chunks.")


# ---------------------------
# 3) Load embeddings model
# ---------------------------
emb_model_hf = HuggingFaceEmbeddings(
    #model_name="all-MiniLM-L6-v2"     # 384 dim: fast, small
    model_name="all-mpnet-base-v2"  # 768 dim: better quality, slower
)


# ---------------------------
# 4) Create / Reset vector DB
# ---------------------------
#adjust
vd_path = (r"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
           r"Processing\datasets\langchain_rag/vector_dbs/chroma_db_polestar_website_us")

# Delete the entire directory if it exists
if os.path.exists(vd_path):
    shutil.rmtree(vd_path)
    print("Existing database directory deleted.")

os.makedirs(vd_path, exist_ok=True)

#adjust
chroma_db = Chroma(
    persist_directory=vd_path,
    embedding_function=emb_model_hf,
    collection_name="polestar_website_database_us",
)


# ---------------------------
# 5) Batch-safe upsert helpers
# ---------------------------
def get_chroma_max_batch_size(db, fallback=5000):
    """
    Chroma has a maximum batch size for upserts (varies by version / backend).
    This tries to read it from a few known internal attributes and falls back safely.
    """
    candidates = [
        lambda: db._collection._client.get_max_batch_size(),
        lambda: db._collection._client.max_batch_size,
        lambda: db._collection._client._server.get_max_batch_size(),
    ]
    for fn in candidates:
        try:
            v = fn()
            if isinstance(v, int) and v > 0:
                return v
        except Exception:
            pass
    return fallback


def make_stable_ids(docs, start_idx=0):
    """
    Optional but recommended: stable IDs prevent duplicates on reruns and make inserts deterministic.
    Uses source/url metadata if present, otherwise 'unknown_source'.
    """
    ids = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source") or d.metadata.get("url") or "unknown_source"
        # Make it filesystem/Chroma friendly (avoid super long IDs if your URLs are huge)
        ids.append(f"{src}::chunk::{start_idx + i}")
    return ids


# ---------------------------
# 6) Add chunks to DB in batches
# ---------------------------
max_bs = get_chroma_max_batch_size(chroma_db, fallback=5000)
print("Chroma max batch size:", max_bs)

for start in tqdm(range(0, len(chunks_web_map), max_bs), desc="Upserting to Chroma"):
    batch = chunks_web_map[start:start + max_bs]
    batch_ids = make_stable_ids(batch, start_idx=start)  # remove this if you don't want custom IDs
    chroma_db.add_documents(documents=batch, ids=batch_ids)

print("Chroma collection created and populated.")

# Count number of chunks in database
count = chroma_db._collection.count()
print("Total chunks:", count)
