import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import re
import json

from png_pdf import pdfContent  # must return extracted text

BASE_URL = "https://iitj.ac.in/"
OUTPUT_JSON = "website_data.json"

visited_links = set()
pdf_links = set()

data = {
    "pages": [],
    "pdfs": []
}

# ------------------ HELPERS ------------------

def is_internal(url):
    return urlparse(url).netloc == urlparse(BASE_URL).netloc


def is_english(url):
    url = url.lower()
    return not (
        "/hi/" in url
        or url.endswith("/hi")
        or "lang=hi" in url
        or "lg=hi" in url
        or "Hindi" in url
        or "hindi" in url
        or "HINDI" in url
    )


def extract_clean_text(soup):
    # remove junk
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    texts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        text = tag.get_text(" ", strip=True)
        if len(text) > 30:
            texts.append(text)

    text = " ".join(texts)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()

# ------------------ CRAWLER ------------------
count =0

def bfs(start_url):
    queue = deque([start_url])

    while queue:
        url = queue.popleft()

        if url in visited_links or not is_internal(url) or not is_english(url):
            continue

        print("🌐 Visiting:", url)
        visited_links.add(url)

        try:
            response = requests.get(url, timeout=6)
            soup = BeautifulSoup(response.text, "html.parser")

            # -------- PAGE TEXT --------
            page_text = extract_clean_text(soup)
            if page_text:
                data["pages"].append({
                    "url": url,
                    "text": page_text
                })

            # -------- LINKS --------
            for tag in soup.find_all("a", href=True):
                link = urljoin(url, tag["href"]).split("#")[0]

                if not is_internal(link) or not is_english(link):
                    continue

                if link.lower().endswith(".pdf"):
                    if link in pdf_links:
                        continue

                    print("📄 PDF found:", link)
                    pdf_links.add(link)

                    try:
                        pdf_text = pdfContent(link)
                        if pdf_text.strip():
                            data["pdfs"].append({
                                "url": link,
                                "text": pdf_text
                            })
                    except Exception as e:
                        print("❌ PDF error:", link, e)

                else:
                    if link not in visited_links:
                        queue.append(link)
                        
                        

            time.sleep(1)

        except Exception as e:
            print("❌ Page error:", url, e)
        
        global count
        count += 1
        if count==20 : break

# ------------------ RUN ------------------

bfs(BASE_URL)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\n✅ Crawl complete")
print("Pages scraped:", len(data["pages"]))
print("PDFs scraped:", len(data["pdfs"]))
print("Saved to:", OUTPUT_JSON)
