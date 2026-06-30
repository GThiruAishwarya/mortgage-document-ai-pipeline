import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.vision_llm import analyze_page
from pipeline.pdf_processor import load_pdf, render_page_as_image
from pipeline.text_cleaner import clean_text

API_KEY = os.environ.get("GROQ_API_KEY", "")
if not API_KEY:
    print("ERROR: GROQ_API_KEY not set. Create a .env file with GROQ_API_KEY=your-key")
    sys.exit(1)

PDF_PATH = "fixtures/loan_doc.pdf"

print(f"Using key: {API_KEY[:8]}...")
print("Loading PDF...")
doc, page_analyses, _ = load_pdf(PDF_PATH)
img = render_page_as_image(doc[0])
pa = page_analyses[0]
pa.raw_text = clean_text(pa.raw_text)

print("Calling Groq vision...")
try:
    result = analyze_page(
        image=img,
        page_num=1,
        total_pages=len(doc),
        raw_text=pa.raw_text,
        api_key=API_KEY,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    print("SUCCESS:", list(result.keys()))
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")