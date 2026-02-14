import pytesseract
from PIL import Image
import requests
from io import BytesIO
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Path to tesseract engine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔗 PDF URL
def pdfContent(pdfUrl,OUTPUT_FILE):

# Download PDF
    headers = {"User-Agent": "Mozilla/5.0"}  # helps if site blocks bots
    response = requests.get(pdfUrl, headers=headers)
    pdf_bytes = io.BytesIO(response.content)

# Open PDF from memory
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]

    # --- Step 1: Try direct text extraction ---
        curr_text = page.get_text().strip()

        if curr_text:
            print(f"Page {page_num+1}: Text layer found ✅")
            full_text += (curr_text + "\n")

        else:
            print(f"Page {page_num+1}: No text layer → Using OCR 🔍")

        # --- Step 2: Convert page to image ---
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

        # --- Step 3: OCR ---
            ocr_text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            full_text += (ocr_text + "\n")

    doc.close()

#print("\n\n========== FINAL EXTRACTED TEXT ==========\n")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write("Source:" + pdfUrl + "\n")
        f.write(full_text + "\n")