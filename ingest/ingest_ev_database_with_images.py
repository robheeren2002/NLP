# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a scraper
# could be deployed. ChatGPT 5.2 was used to help make this code
# work also with images, and JavaScript sites.
# All chats can be found at:
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# https://chatgpt.com/share/699083b7-0c58-8013-9c8b-c70222e9d519
# ============================================================
import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
car_brand = "Volvo"

SITEMAP_URL = "https://ev-database.org/sitemap.xml"
FILTER_URLS = [rf"^https://ev-database\.org/car/\d+/{re.escape(car_brand)}.*$"]

REQUESTS_PER_SECOND = 0.1
REQUEST_SLEEP = 1.0 / max(REQUESTS_PER_SECOND, 1e-9)

OUTPUT_DIR = (
    r"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
    r"Processing\datasets\langchain_rag\backup"
)
OUTPUT_FILENAME = f"docs_web_map_ev_database_{car_brand}_with_images.json"

# Image extraction controls
MAX_IMAGES_PER_PAGE = 25
SKIP_IMAGE_SUBSTRINGS = ["sprite", "icon", "logo"]  # tune as needed

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)
TIMEOUT_SECONDS = 30


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def save_to_local(docs: List[Document], directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Successfully saved {len(docs)} docs to {full_path}")


def fetch_html(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.text


def fetch_sitemap_urls(sitemap_url: str) -> List[str]:
    """
    Fetch sitemap.xml and extract all <loc> URLs.
    """
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(sitemap_url, headers=headers, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc") if loc.get_text(strip=True)]
    return locs


def filter_urls(urls: List[str], patterns: List[str]) -> List[str]:
    regs = [re.compile(p) for p in patterns]
    out = []
    for u in urls:
        if any(rx.search(u) for rx in regs):
            out.append(u)
    return out


def html_to_text(html: str) -> str:
    """
    Turn HTML into readable-ish plain text.
    Keeps the spirit of your previous output (lots of content),
    but removes scripts/styles and collapses some whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious non-content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n", strip=False)

    # Normalize line endings a bit (keep it verbose, like your samples)
    # - collapse super-long whitespace runs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def pick_best_from_srcset(srcset: str) -> Optional[str]:
    """
    srcset example: "a.jpg 480w, b.jpg 800w, c.jpg 1200w"
    Picks the largest width candidate.
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
                except Exception:
                    width = 0
        candidates.append((width, url))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def should_skip_image(url: str) -> bool:
    u = (url or "").lower()
    return any(s in u for s in SKIP_IMAGE_SUBSTRINGS)


def extract_images_from_html(
    html: str,
    base_url: str,
    max_images: int = MAX_IMAGES_PER_PAGE,
) -> List[Dict[str, Any]]:
    """
    Extract image references from HTML via BeautifulSoup.
    Handles:
      - <img src / data-src / data-lazy-src / srcset / data-srcset>
      - <picture><source srcset=...>
      - meta og:image / twitter:image / itemprop=image
      - inline style background-image:url(...)
    Returns list of dicts: [{"url": abs_url, "alt": "..."}]
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    images: List[Dict[str, Any]] = []
    seen = set()

    def add(url: str, alt: str = ""):
        if not url:
            return
        abs_url = urljoin(base_url, url)
        if not abs_url or abs_url in seen:
            return
        if should_skip_image(abs_url):
            return
        seen.add(abs_url)
        images.append({"url": abs_url, "alt": (alt or "").strip()})

    # 1) <img> tags
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        srcset = img.get("srcset") or img.get("data-srcset")
        best_srcset = pick_best_from_srcset(srcset) if srcset else None
        chosen = best_srcset or src
        add(chosen, img.get("alt") or "")
        if len(images) >= max_images:
            return images

    # 2) <picture><source srcset=...> (often the real asset is here)
    for source in soup.find_all("source"):
        srcset = source.get("srcset")
        best = pick_best_from_srcset(srcset) if srcset else None
        add(best, "")
        if len(images) >= max_images:
            return images

    # 3) Social/meta image hints
    meta_selectors = [
        ("meta", {"property": "og:image"}),
        ("meta", {"name": "twitter:image"}),
        ("meta", {"itemprop": "image"}),
    ]
    for tag, attrs in meta_selectors:
        for m in soup.find_all(tag, attrs=attrs):
            add(m.get("content"), "")
            if len(images) >= max_images:
                return images

    # 4) Inline CSS background-image:url(...)
    for el in soup.select('[style*="background-image"]'):
        style = el.get("style") or ""
        m = re.search(
            r'background-image\s*:\s*url\((["\']?)(.*?)\1\)',
            style,
            re.IGNORECASE,
        )
        if m:
            add(m.group(2), "")
            if len(images) >= max_images:
                return images

    return images


def scrape_ev_database_pages(urls: List[str]) -> List[Document]:
    docs: List[Document] = []
    total = len(urls)

    for i, url in enumerate(urls, start=1):
        print(f"📄 Scraping {i}/{total}: {url}")

        # polite throttling
        time.sleep(REQUEST_SLEEP)

        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"⚠️  Failed to fetch {url}: {e}")
            doc = Document(
                page_content="",
                metadata={
                    "source": url,
                    "loc": url,
                    "tag_source": "ev_database",
                    "brand": car_brand,
                    "images": [],
                },
            )
            docs.append(doc)
            continue

        page_text = html_to_text(html)

        images = extract_images_from_html(html, url, max_images=MAX_IMAGES_PER_PAGE)
        for im in images:
            im["source"] = url  # required in your output structure

        doc = Document(
            page_content=page_text,
            metadata={
                "source": url,
                "loc": url,
                "tag_source": "ev_database",
                "brand": car_brand,
                "images": images,
            },
        )
        docs.append(doc)

        if i % 25 == 0:
            print(f"🖼️  Processed {i}/{total} pages...")

    return docs


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
print("🔎 Fetching sitemap URLs...")
all_urls = fetch_sitemap_urls(SITEMAP_URL)
print(f"📌 Sitemap URLs found: {len(all_urls)}")

print("🎯 Filtering vehicle URLs...")
vehicle_urls = filter_urls(all_urls, FILTER_URLS)
print(f"🚗 Matching vehicle URLs: {len(vehicle_urls)}")

print("🧾 Scraping pages + images (static HTML)...")
docs_final = scrape_ev_database_pages(vehicle_urls)

print(f"📦 Final document count: {len(docs_final)}")

print("💾 Saving JSON...")
save_to_local(
    docs=docs_final,
    directory=OUTPUT_DIR,
    filename=OUTPUT_FILENAME,
)

print("🚀 Static scraping pipeline completed successfully.")
