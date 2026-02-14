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
import nest_asyncio
from typing import List
from langchain_community.document_loaders import PlaywrightURLLoader, SitemapLoader
from langchain_core.documents import Document

nest_asyncio.apply()  # allow async in notebooks

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SITEMAP_URL = "https://www.polestar.com/sitemap.xml"
FILTER_URLS = ["https://www.polestar.com/us/"]

REQUESTS_PER_SECOND = 2

OUTPUT_DIR = (
    r"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
    r"Processing\datasets\langchain_rag\backup"
)
OUTPUT_FILENAME = "docs_web_map_polestar_us.json"

JS_FALLBACK_MARKERS = [
    "JavaScript seems disabled",
    "enable JavaScript",
    "This application require JavaScript",
]

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def is_js_fallback(doc: Document) -> bool:
    """Detect pages that failed due to missing JS rendering."""
    content = doc.page_content.lower()
    return any(marker.lower() in content for marker in JS_FALLBACK_MARKERS)


def save_to_local(docs: List[Document], directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)

    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Successfully saved {len(docs)} docs to {full_path}")


# ------------------------------------------------------------
# STEP 1 — LOAD VIA SITEMAP (FAST)
# ------------------------------------------------------------
if True:
    print("🔎 Loading pages via SitemapLoader...")

    loader_web_map = SitemapLoader(
        web_path=SITEMAP_URL,
        filter_urls=FILTER_URLS,
    )

    loader_web_map.requests_per_second = REQUESTS_PER_SECOND

    docs_sitemap = loader_web_map.load()
    print(f"📄 Loaded {len(docs_sitemap)} document(s) from sitemap")

    # add metadata
    for doc in docs_sitemap:
        doc.metadata["tag_source"] = "website_polestar"
        doc.metadata["locale"] = "nl-be"
else:
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
    docs_sitemap  = load_from_local(
    directory=r"C:\Users\robhe\OneDrive - Vlerick Business School\Natural Language "
              r"Processing\datasets\langchain_rag\backup",
    filename="docs_web_map_polestar.json"
)

# ------------------------------------------------------------
# STEP 2 — DETECT JS-ONLY PAGES
# ------------------------------------------------------------

docs_js_fallback = [d for d in docs_sitemap if is_js_fallback(d)]
docs_ok = [d for d in docs_sitemap if not is_js_fallback(d)]

print(f"⚠️  JS fallback pages detected: {len(docs_js_fallback)}")
print(f"✅ Clean static pages: {len(docs_ok)}")

# ------------------------------------------------------------
# STEP 3 — RELOAD JS PAGES WITH PLAYWRIGHT
# ------------------------------------------------------------
from typing import List
from langchain_core.documents import Document
from playwright.sync_api import sync_playwright
import time


class CookieAwarePlaywrightLoader:
    def __init__(self, urls: List[str], headless: bool = True):
        self.urls = urls
        self.headless = headless

    def load(self) -> List[Document]:
        documents = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            print("Loading pages...: 0/", len(self.urls))
            count = 0
            for url in self.urls:
                count += 1
                print("Loading pages...: ",count,"/", len(self.urls))
                page = context.new_page()
                page.goto(url, timeout=60000)

                # --------------------------------------------------
                # ACCEPT COOKIES (OneTrust – Polestar)
                # --------------------------------------------------
                try:
                    page.wait_for_selector(
                        "button:has-text('ccept')",
                        timeout=5000,
                    )
                    page.click("button:has-text('ccept')")
                    time.sleep(1)
                except:
                    pass  # banner not present

                # --------------------------------------------------
                # WAIT FOR MAIN CONTENT TO LOAD
                # --------------------------------------------------
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except:
                    pass

                # Optional: scroll to trigger lazy loading
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)

                content = page.inner_text("body")

                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": url},
                    )
                )

                page.close()

            browser.close()

        return documents

docs_js_rendered = []

if docs_js_fallback:
    urls_to_reload = [d.metadata["source"] for d in docs_js_fallback]

    print("🧠 Reloading JS pages with Playwright...")
    print(f"🌐 URLs: {len(urls_to_reload)}")

    loader_js = CookieAwarePlaywrightLoader(
        urls=urls_to_reload,
        headless=True,
    )

    docs_js_rendered = loader_js.load()

    for doc in docs_js_rendered:
        doc.metadata["tag_source"] = "website_polestar"
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

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------

print("🚀 Hybrid loading pipeline completed successfully.")
