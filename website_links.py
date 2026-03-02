
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import re
import json

from pdf_processing import f_pdf_content  # must return extracted text

BASE_URL = "https://iitj.ac.in/"
OUTPUT_JSON = "website_data.json"

visited_links = set()
pdf_links = set()

data = {
    "pages": [],
    "pdfs": []
}

# ------------------ HELPERS ------------------

def f_is_internal(url):
    return urlparse(url).netloc == urlparse(BASE_URL).netloc


def f_is_english(url):
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


def f_extract_structured_content(soup):
    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    structured_data = []
    current_heading = "General"

    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = tag.get_text(" ", strip=True)

        # Skip tiny noise text
        if len(text) < 30:
            continue

        # Clean whitespace + remove non-ascii
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)

        if tag.name in ["h1", "h2", "h3"]:
            current_heading = text
        else:
            structured_data.append({
                "heading": current_heading,
                "content": text
            })

    return structured_data

# ------------------ CRAWLER ------------------
count =0

def f_bfs(start_url):
    queue = deque([start_url])

    while queue:
        url = queue.popleft()

        if url in visited_links or not f_is_internal(url) or not f_is_english(url):
            continue

        #print("🌐 Visiting:", url)
        visited_links.add(url)

        try:
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, "html.parser")

            # -------- PAGE TEXT --------
            # -------- PAGE TEXT --------
            page_title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Page"

            structured_content = f_extract_structured_content(soup)

            if structured_content:
                data["pages"].append({
                    "url": url,
                    "title": page_title,
                    "sections": structured_content
                })

            # -------- LINKS --------
            for tag in soup.find_all("a", href=True):
                link = urljoin(url, tag["href"]).split("#")[0]

                if not f_is_internal(link) or not f_is_english(link):
                    continue

                if link.lower().endswith(".pdf"):
                    if link in pdf_links:
                        continue

                    #print("📄 PDF found:", link)
                    pdf_links.add(link)

                    try:
                        pdf_text = f_pdf_content(link)
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
                        
                        

            time.sleep(0.1)

        except Exception as e:
            print("❌ Page error:", url, e)
        
        global count
        count += 1
        print(count)
        

# ------------------ RUN ------------------

f_bfs(BASE_URL)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\n✅ Crawl complete")
print("Pages scraped:", len(data["pages"]))
print("PDFs scraped:", len(data["pdfs"]))
print("Saved to:", OUTPUT_JSON)

"""
After the entire website is crawled and data is extracted the vector store needs to be refreshed as well.
"""
import subprocess
subprocess.run(["python", "vector_store.py"])
