# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a scraper
# could be deployed. ChatGPT 5.2 was used to help make this code
# work also with images, and JavaScript sites.
# All chats can be found at:
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# ============================================================

import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import SitemapLoader
from langchain_core.documents import Document

from playwright.sync_api import sync_playwright

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
car_brand = 'BMW'
SITEMAP_URL = "https://ev-database.org/sitemap.xml"
FILTER_URLS = [rf"^https://ev-database\.org/car/\d+/{re.escape(car_brand)}.*$"]


REQUESTS_PER_SECOND = 0.1
REQUEST_SLEEP = 1.0 / max(REQUESTS_PER_SECOND, 1e-9)

OUTPUT_DIR = (
    r"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
    r"Processing\datasets\langchain_rag\backup"
)
OUTPUT_FILENAME = f"docs_web_map_ev_database_{car_brand}_with_images.json"

JS_FALLBACK_MARKERS = [
    "JavaScript seems disabled",
    "enable JavaScript",
    "This application require JavaScript",
    # EV-Database WAF block page
    "request blocked",
    "anomalies detected",
    "block-",
]

# Image extraction controls
MAX_IMAGES_PER_PAGE = 25
SKIP_IMAGE_SUBSTRINGS = ["sprite", "icon", "logo"]  # tune as needed
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
TIMEOUT_SECONDS = 30


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def is_js_fallback(doc: Document) -> bool:
    """Detect pages that failed due to missing JS rendering."""
    content = (doc.page_content or "").lower()
    return any(marker.lower() in content for marker in JS_FALLBACK_MARKERS)


def save_to_local(docs: List[Document], directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Successfully saved {len(docs)} docs to {full_path}")


def _pick_best_from_srcset(srcset: str) -> Optional[str]:
    """
    srcset example: "a.jpg 480w, b.jpg 800w, c.jpg 1200w"
    We pick the largest width candidate.
    """
    if not srcset:
        return None
    candidates = []
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        toks = part.split()
        url = toks[0].strip()
        width = 0
        if len(toks) > 1:
            w = toks[1].strip().lower()
            if w.endswith("w"):
                try:
                    width = int(w[:-1])
                except:
                    width = 0
        candidates.append((width, url))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _should_skip_image(url: str) -> bool:
    u = (url or "").lower()
    return any(s in u for s in SKIP_IMAGE_SUBSTRINGS)


def extract_images_from_html(html: str, base_url: str, max_images: int = MAX_IMAGES_PER_PAGE) -> List[Dict[str, Any]]:
    """
    Extract image references from HTML via BeautifulSoup.
    Handles src, data-src, data-lazy-src, and srcset (chooses largest).
    Returns list of dicts: [{"url": abs_url, "alt": "..."}]
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    images: List[Dict[str, Any]] = []
    seen = set()

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        srcset = img.get("srcset") or img.get("data-srcset")
        best_srcset = _pick_best_from_srcset(srcset) if srcset else None

        chosen = best_srcset or src
        if not chosen:
            continue

        abs_url = urljoin(base_url, chosen)
        if not abs_url or abs_url in seen:
            continue
        if _should_skip_image(abs_url):
            continue

        seen.add(abs_url)
        images.append({
            "url": abs_url,
            "alt": (img.get("alt") or "").strip()
        })

        if len(images) >= max_images:
            break

    return images


def fetch_html(url: str) -> str:
    """Fetch raw HTML for non-JS pages (fallback if SitemapLoader output isn't HTML)."""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.text


def attach_images_to_docs_sitemap(docs: List[Document]) -> List[Document]:
    """
    For each sitemap-loaded doc:
    - Try extracting images from doc.page_content if it looks like HTML
    - If not HTML, optionally fetch HTML and extract (slower but robust)
    """
    out = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "")
        if not src:
            d.metadata["images"] = []
            out.append(d)
            continue

        html_candidate = d.page_content or ""
        images = []

        # Fast path: if content seems to contain HTML image tags
        if "<img" in html_candidate.lower() or "<html" in html_candidate.lower():
            images = extract_images_from_html(html_candidate, src)

        # Robust fallback: fetch HTML if we got no images and content isn't HTML
        if not images:
            try:
                time.sleep(REQUEST_SLEEP)
                html = fetch_html(src)
                images = extract_images_from_html(html, src)
            except Exception:
                images = []

        # Store in metadata (and include page url on each image)
        for im in images:
            im["source"] = src

        d.metadata["images"] = images
        out.append(d)

        if i % 25 == 0:
            print(f"🖼️  Processed images for {i}/{len(docs)} sitemap docs...")

    return out


def extract_images_from_playwright(page, base_url: str, max_images: int = MAX_IMAGES_PER_PAGE) -> List[Dict[str, Any]]:
    """
    Extract images from the live DOM in Playwright.
    Also checks srcset, data-src, data-lazy-src, data-srcset.
    """
    try:
        raw = page.eval_on_selector_all(
            "img",
            """els => els.map(e => ({
                src: e.getAttribute('src') || '',
                alt: e.getAttribute('alt') || '',
                dataSrc: e.getAttribute('data-src') || '',
                dataLazy: e.getAttribute('data-lazy-src') || '',
                srcset: e.getAttribute('srcset') || '',
                dataSrcset: e.getAttribute('data-srcset') || ''
            }))"""
        )
    except Exception:
        return []

    images: List[Dict[str, Any]] = []
    seen = set()

    for it in raw:
        src = it.get("src") or it.get("dataSrc") or it.get("dataLazy")
        srcset = it.get("srcset") or it.get("dataSrcset")
        best_srcset = _pick_best_from_srcset(srcset) if srcset else None
        chosen = best_srcset or src
        if not chosen:
            continue

        abs_url = urljoin(base_url, chosen)
        if not abs_url or abs_url in seen:
            continue
        if _should_skip_image(abs_url):
            continue

        seen.add(abs_url)
        images.append({
            "url": abs_url,
            "alt": (it.get("alt") or "").strip(),
            "source": base_url
        })

        if len(images) >= max_images:
            break

    return images


# ------------------------------------------------------------
# PLAYWRIGHT LOADER (COOKIE AWARE) + IMAGES
# ------------------------------------------------------------

class CookieAwarePlaywrightLoader:
    def __init__(self, urls: List[str], headless: bool = True):
        self.urls = urls
        self.headless = headless

    def load(self) -> List[Document]:
        documents: List[Document] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(user_agent=USER_AGENT)
            print("Loading pages...: 0/", len(self.urls))

            for idx, url in enumerate(self.urls, start=1):
                print("Loading pages...: ", idx, "/", len(self.urls))
                page = context.new_page()

                try:
                    page.goto(url, timeout=60000)
                except Exception:
                    # If navigation fails, store empty content but keep source
                    documents.append(Document(page_content="", metadata={"source": url, "images": []}))
                    page.close()
                    continue

                # --------------------------------------------------
                # ACCEPT COOKIES (best-effort)
                # --------------------------------------------------
                try:
                    page.wait_for_selector("button:has-text('ccept')", timeout=5000)
                    page.click("button:has-text('ccept')")
                    time.sleep(1)
                except:
                    pass

                # --------------------------------------------------
                # WAIT FOR MAIN CONTENT TO LOAD
                # --------------------------------------------------
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except:
                    pass

                # Trigger lazy loading
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(1)
                except:
                    pass

                # TEXT for RAG
                try:
                    content_text = page.inner_text("body")
                except Exception:
                    content_text = ""

                # IMAGE REFERENCES
                images = extract_images_from_playwright(page, url, max_images=MAX_IMAGES_PER_PAGE)

                documents.append(
                    Document(
                        page_content=content_text,
                        metadata={
                            "source": url,
                            "images": images,
                        },
                    )
                )

                page.close()

            browser.close()

        return documents


# ------------------------------------------------------------
# STEP 1 — LOAD VIA SITEMAP (FAST)
# ------------------------------------------------------------

print("🔎 Loading pages via SitemapLoader...")

loader_web_map = SitemapLoader(
    web_path=SITEMAP_URL,
    filter_urls=FILTER_URLS,
    requests_per_second=REQUESTS_PER_SECOND,
)

docs_sitemap = loader_web_map.load()

print(f"📄 Loaded {len(docs_sitemap)} document(s) from sitemap")

# Tag base metadata
for doc in docs_sitemap:
    doc.metadata["tag_source"] = "ev_database"
    doc.metadata["brand"] = car_brand

# Attach images to sitemap docs
print("🖼️  Extracting image references for sitemap docs...")
docs_sitemap = attach_images_to_docs_sitemap(docs_sitemap)


# ------------------------------------------------------------
# STEP 2 — DETECT JS-ONLY PAGES
# ------------------------------------------------------------

docs_js_fallback = [d for d in docs_sitemap if is_js_fallback(d)]
docs_ok = [d for d in docs_sitemap if not is_js_fallback(d)]

print(f"⚠️  JS fallback pages detected: {len(docs_js_fallback)}")
print(f"✅ Clean static pages: {len(docs_ok)}")


# ------------------------------------------------------------
# STEP 3 — RELOAD JS PAGES WITH PLAYWRIGHT (and images)
# ------------------------------------------------------------

docs_js_rendered: List[Document] = []

if docs_js_fallback:
    urls_to_reload = [d.metadata["source"] for d in docs_js_fallback if d.metadata.get("source")]

    print("🧠 Reloading JS pages with Playwright...")
    print(f"🌐 URLs: {len(urls_to_reload)}")

    loader_js = CookieAwarePlaywrightLoader(
        urls=urls_to_reload,
        headless=True,
    )

    docs_js_rendered = loader_js.load()

    for doc in docs_js_rendered:
        doc.metadata["tag_source"] = "website_js_rendered"
        doc.metadata["locale"] = "nl-be"

    print(f"🎉 JS-rendered pages loaded: {len(docs_js_rendered)}")


# ------------------------------------------------------------
# STEP 4 — MERGE RESULTS
# ------------------------------------------------------------

docs_final = docs_ok + docs_js_rendered
print(f"📦 Final document count: {len(docs_final)}")


# ------------------------------------------------------------
# STEP 5 — SAVE TO JSON
# ------------------------------------------------------------

save_to_local(
    docs=docs_final,
    directory=OUTPUT_DIR,
    filename=OUTPUT_FILENAME,
)

print("🚀 Hybrid loading pipeline completed successfully.")
