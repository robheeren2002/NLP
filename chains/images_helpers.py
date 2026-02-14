# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a RAG chain
# could be invoked. ChatGPT 5.2 was used to help make this code
# more modular, and to help with integrating images into the RAG.
# All chats can be found at:
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# ============================================================
import json
import math
import os
import re
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings

from inputs import vd_path, IMAGE_INDEX_PATH

# Embeddings model must match the DB embedding model
emb_model_hf = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

with open(IMAGE_INDEX_PATH, "r", encoding="utf-8") as f:
    IMAGE_INDEX = json.load(f)

def user_asked_for_images(question: str) -> bool:
    q = (question or "").lower()
    triggers = [
        "image", "images", "picture", "pictures", "photo", "photos",
        "show me", "show", "what does it look like", "look like"
    ]
    return any(t in q for t in triggers)

def cosine(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def image_label(abs_path: str, alt: str, source: str, url: str) -> str:
    fname = os.path.basename(abs_path)
    parts = [fname]
    if alt:
        parts.append(alt)
    if source:
        parts.append(source)
    if url:
        parts.append(url)
    return " | ".join(parts)

def best_image_per_source_semantic(docs, question: str, max_sources=10):
    sources = []
    seen_src = set()
    for d in docs:
        src = d.metadata.get("source", "N/A")
        if src != "N/A" and src not in seen_src:
            sources.append(src)
            seen_src.add(src)
        if len(sources) >= max_sources:
            break

    if not sources:
        return {}

    q_vec = emb_model_hf.embed_query(question)

    out = {}
    for src in sources:
        image_key = None
        for d in docs:
            if d.metadata.get("source") == src and d.metadata.get("image_key"):
                image_key = d.metadata["image_key"]
                break
        if not image_key:
            continue

        imgs = IMAGE_INDEX.get(image_key, [])
        if not imgs:
            continue

        best = None
        best_score = -999.0

        for im in imgs:
            rel = im.get("path")
            if not rel:
                continue

            abs_path = os.path.join(vd_path, rel)
            if not os.path.exists(abs_path):
                continue

            alt = (im.get("alt") or "").strip()
            url = (im.get("url") or "").strip()
            label = image_label(abs_path, alt, src, url)
            v = emb_model_hf.embed_query(label)
            score = cosine(q_vec, v)

            if score > best_score:
                best_score = score
                best = {
                    "abs_path": abs_path,
                    "rel_path": rel.replace("\\", "/"),
                    "alt": alt,
                    "source_id": src,
                    "url": url,
                    "sha256": sha256_file(abs_path),
                }

        if best:
            out[src] = best

    return out

def build_image_context_from_map(image_map: dict) -> str:
    if not image_map:
        return "None"
    lines = []
    for src, im in image_map.items():
        fname = os.path.basename(im.get("abs_path", ""))
        alt = (im.get("alt") or "").strip()
        if alt:
            lines.append(f"- Source_id: {src} -> {fname} (alt: {alt})")
        else:
            lines.append(f"- Source_id: {src} -> {fname}")
    return "\n".join(lines)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Accept both:
#  - Source_id: https://...
#  - Source_id: [label](https://...)
SOURCE_ID_PLAIN_RE = re.compile(r"Source_id:\s*(https?://\S+)", re.IGNORECASE)
SOURCE_ID_MD_RE = re.compile(r"Source_id:\s*\[[^\]]+\]\((https?://[^)]+)\)", re.IGNORECASE)

def extract_cited_source_ids(answer_text: str) -> list[str]:
    if not answer_text:
        return []

    cites = []
    cites.extend(SOURCE_ID_MD_RE.findall(answer_text))
    cites.extend(SOURCE_ID_PLAIN_RE.findall(answer_text))

    cleaned = []
    for c in cites:
        c = c.strip().rstrip(").,]")
        cleaned.append(c)

    seen = set()
    out = []
    for c in cleaned:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def normalize_source_id_citations_to_plain(answer_text: str) -> str:
    """
    Convert 'Source_id: [x](url)' -> 'Source_id: url' (so your regex & image linking are stable)
    """
    return SOURCE_ID_MD_RE.sub(lambda m: f"Source_id: {m.group(1)}", answer_text or "")
