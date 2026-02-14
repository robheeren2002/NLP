# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code from class to create
# a chroma DB. ChatGPT 5.2 was used to help make this code
# work with images aswel.
# All chats can be found at:
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# ============================================================
import os
import json
import shutil
import hashlib
from urllib.parse import urlparse
from tqdm import tqdm

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------
# CONFIG (ADJUST)
# ---------------------------
car_brand = 'combined'
JSON_DIR = (r"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
            r"Processing\datasets\langchain_rag\backup")
JSON_FILE = f"combined.json"
VD_PATH = (fr"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
           fr"Processing\datasets\langchain_rag/vector_dbs/chroma_db_ev_database_{car_brand}")

COLLECTION_NAME = f"ev_database_{car_brand}"

# Store images next to the DB
IMAGES_DIR = os.path.join(VD_PATH, "images")
IMAGE_INDEX_PATH = os.path.join(VD_PATH, "image_index.json")

MAX_IMAGES_PER_PAGE = 10
TIMEOUT_SECONDS = 25
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
SKIP_SUBSTRINGS = ["sprite", "icon", "logo"]


# ---------------------------
# 1) Load docs from JSON
# ---------------------------
def load_from_local(directory, filename):
    full_path = os.path.join(directory, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    docs = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in loaded_data]
    print(f"Loaded {len(docs)} documents from {full_path}")
    return docs


# ---------------------------
# 2) Offline image download + index build
# ---------------------------
def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _safe_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return ext
    return ".jpg"


def _is_skippable(url: str) -> bool:
    u = (url or "").lower()
    return any(s in u for s in SKIP_SUBSTRINGS)


def download_image(url: str, out_dir: str) -> str | None:
    """
    Download image to out_dir using stable hashed filename.
    Returns absolute local path or None.
    """
    if not url or _is_skippable(url):
        return None

    os.makedirs(out_dir, exist_ok=True)
    ext = _safe_ext_from_url(url)
    fname = f"{_hash(url)}{ext}"
    fpath = os.path.join(out_dir, fname)

    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        return fpath

    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS, stream=True)
        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return None

        with open(fpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)

        if os.path.getsize(fpath) < 1024:
            try:
                os.remove(fpath)
            except:
                pass
            return None

        return fpath
    except Exception:
        return None


def build_offline_image_index_and_simplify_metadata(
    docs: list[Document],
    images_dir_abs: str,
    base_dir_abs: str,
    index_path: str
) -> tuple[list[Document], dict]:
    """
    Creates image_index.json keyed by a stable image_key (string).
    Replaces doc.metadata["images"] (complex list) with:
      - doc.metadata["image_key"] : str
      - doc.metadata["has_images"]: bool
    This makes Chroma happy (primitive-only metadata).
    """
    os.makedirs(images_dir_abs, exist_ok=True)

    image_index: dict[str, list[dict]] = {}

    # estimate
    est = sum(min(len(d.metadata.get("images", []) or []), MAX_IMAGES_PER_PAGE) for d in docs)
    print(f"🖼️  Will attempt up to ~{est} image downloads (cap/page={MAX_IMAGES_PER_PAGE}).")

    for d in tqdm(docs, desc="Downloading images + building index"):
        src = d.metadata.get("source") or d.metadata.get("loc") or ""
        # stable key per page
        image_key = _hash(src) if src else _hash(d.page_content[:200])

        imgs = d.metadata.get("images") or []
        offline_imgs = []

        if isinstance(imgs, list) and imgs:
            for im in imgs[:MAX_IMAGES_PER_PAGE]:
                url = (im.get("url") or "").strip()
                alt = (im.get("alt") or "").strip()

                local_abs = download_image(url, images_dir_abs)
                if not local_abs:
                    continue

                rel_path = os.path.relpath(local_abs, start=base_dir_abs).replace("\\", "/")

                offline_imgs.append({
                    "path": rel_path,
                    "alt": alt,
                    "url": url,
                    "source": src
                })

        # store in side index if any
        if offline_imgs:
            image_index[image_key] = offline_imgs

        # IMPORTANT: remove complex metadata
        if "images" in d.metadata:
            del d.metadata["images"]

        # store only primitive pointers
        d.metadata["image_key"] = image_key
        d.metadata["has_images"] = bool(offline_imgs)

    # save index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(image_index, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved image index: {index_path} (keys={len(image_index)})")
    return docs, image_index


# ---------------------------
# 3) Split docs (unchanged)
# ---------------------------
splitter_web = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024,
    chunk_overlap=128,
    separators=["\n\n", "\n", " ", ""]
)


# ---------------------------
# 4) Embeddings (unchanged)
# ---------------------------
emb_model_hf = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


# ---------------------------
# 5) Chroma helpers (unchanged)
# ---------------------------
def get_chroma_max_batch_size(db, fallback=5000):
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
    ids = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source") or d.metadata.get("url") or d.metadata.get("loc") or "unknown_source"
        ids.append(f"{src}::chunk::{start_idx + i}")
    return ids


# ---------------------------
# RUN
# ---------------------------
docs = load_from_local(JSON_DIR, JSON_FILE)
if not docs:
    raise SystemExit("No documents loaded. Exiting.")

# Reset bundle folder (DB + images + image_index.json)
if os.path.exists(VD_PATH):
    shutil.rmtree(VD_PATH)
    print("Existing database directory deleted.")
os.makedirs(VD_PATH, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Build offline image index + simplify metadata for Chroma
docs, _image_index = build_offline_image_index_and_simplify_metadata(
    docs=docs,
    images_dir_abs=IMAGES_DIR,
    base_dir_abs=VD_PATH,
    index_path=IMAGE_INDEX_PATH
)

# Split AFTER simplifying so chunks inherit primitive metadata only
chunks = splitter_web.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# Create DB
chroma_db = Chroma(
    persist_directory=VD_PATH,
    embedding_function=emb_model_hf,
    collection_name=COLLECTION_NAME,
)

# Upsert
max_bs = get_chroma_max_batch_size(chroma_db, fallback=5000)
print("Chroma max batch size:", max_bs)

for start in tqdm(range(0, len(chunks), max_bs), desc="Upserting to Chroma"):
    batch = chunks[start:start + max_bs]
    batch_ids = make_stable_ids(batch, start_idx=start)
    chroma_db.add_documents(documents=batch, ids=batch_ids)

print("✅ Chroma collection created and populated.")
print("Total chunks:", chroma_db._collection.count())
print(f"📦 Offline bundle:\n  {VD_PATH}\n  images/: {IMAGES_DIR}\n  image index: {IMAGE_INDEX_PATH}")
