import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from collections import deque
import re
from png_pdf import pdfContent

BASE_URL = "https://iitj.ac.in/"

visited_links = set()
pdf_links = set()
png_links = set()

OUTPUT_FILE = "website_content.txt"


def is_internal(url):
    return urlparse(url).netloc == urlparse(BASE_URL).netloc


def is_english(url):
    url = url.lower()
    return not (
        "/hi/" in url
        or url.endswith("/hi")
        or "lang=hi" in url
        or "lg=hi" in url
        or url.endswith("hi")
    )


def extract_clean_text(soup):
    """Remove junk and extract meaningful content"""

    # ❌ Remove layout + non-content sections
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    # ✅ Only keep meaningful content tags
    content_tags = soup.find_all(["p", "h1", "h2", "h3", "li"])

    texts = []
    for tag in content_tags:
        text = tag.get_text(" ", strip=True)

        # Remove very small or useless fragments
        if len(text) > 30:
            texts.append(text)

    page_text = "\n".join(texts)

    # 🧼 Cleaning
    page_text = re.sub(r"\s+", " ", page_text)  # normalize whitespace
    page_text = re.sub(r"[^\x00-\x7F]+", " ", page_text)  # remove weird unicode

    return page_text.strip()


def bfs(start_url):
    queue = deque([start_url])

    while queue:
        url = queue.popleft()

        if url in visited_links or not is_internal(url) or not is_english(url):
            continue

        print("Visiting page:", url)
        visited_links.add(url)

        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")

            # -------- CLEAN TEXT EXTRACTION --------
            page_text = extract_clean_text(soup)

            if page_text:  # avoid saving empty pages
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"Source: {url}\n\n")
                    f.write(page_text + "\n")

            # -------- LINK CHECK --------
            for tag in soup.find_all("a", href=True):
                link = urljoin(url, tag["href"]).split("#")[0]

                if not is_internal(link) or not is_english(link):
                    continue

                if link.lower().endswith(".pdf"):
                    if link not in pdf_links:
                        pdf_links.add(link)
                        print("PDF found:", link)
                        pdfContent(link, OUTPUT_FILE)
                else:
                    if link not in visited_links:
                        queue.append(link)

            time.sleep(1)

        except Exception as e:
            print("Error at", url, ":", e)


# Clear old file
open(OUTPUT_FILE, "w", encoding="utf-8").close()

# Start crawl
bfs(BASE_URL)

print("\nTotal pages visited:", len(visited_links))
print("Total PDFs found:", len(pdf_links))
print("Total PNGs found:", len(png_links))
print("All data saved to:", OUTPUT_FILE)
