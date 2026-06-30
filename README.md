Mortgage Document AI Pipeline
FinacPlus AI Engineering Intern — Take-Home Assignment (Part 1)

Time spent: ~28 hours

What this does
A production-grade extraction pipeline that processes mortgage PDFs (Loan Estimates, Appraisals, Closing Disclosures, Promissory Notes, Deeds of Trust, HUD-1s) and returns structured JSON with:

Every field carrying source_page, confidence (high/medium/low), a one-line reason when not high, and the exact evidence string
Explicit, auditable correction resolution — not a "last-page wins" assumption
Arithmetic mismatch detection — flagged and preserved as-stated, never silently recomputed
Table extraction (both native text-layer via pdfplumber and vision-based via Groq)
Stamp, signature, watermark, and handwriting detection
Degraded text block extraction with partial content and cause
Context-aware document classification (hybrid rule-based + LLM)
Disk-based result cache — re-runs cost $0 for already-processed pages
Requirements
Python 3.11+
A Groq API key (free tier works — get one at https://console.groq.com)
Quick start
# 1. Unzip / clone the repo
cd mortgage-pipeline
# 2. Install dependencies
pip install -r requirements.txt
# 3. Set your Groq API key
#    Linux / macOS
export GROQ_API_KEY="your-key-here"
#    Windows CMD
set GROQ_API_KEY=your-key-here
#    Or create a .env file in the project root:
#    GROQ_API_KEY=your-key-here
# 4. Run the Streamlit app
streamlit run app.py --server.port 8000

Open http://localhost:8000 in your browser.
Upload one or both PDFs from fixtures/ and click Run Pipeline.

Cache management
Results are cached in .llm_cache/ by page-image hash. To force a fresh run:

# Linux / macOS
rm -rf .llm_cache
# Windows CMD
rmdir /s /q .llm_cache

Fixture documents
File	Type	Intentional challenges
fixtures/loan_doc.pdf	Loan Estimate	Fee correction addendum ($9,920 → $11,475), degraded signature block, OCR-noisy loan number
fixtures/appraisal_doc.pdf	Property Appraisal Summary	Appraised value revised downward on addendum ($612,000 → $586,500), rotated stamp, partial reviewer text
Output schema
Every run produces an ExtractionResult (also exportable as JSON from the UI):

{
  "document_id": "abc12345",
  "filename": "loan_doc.pdf",
  "document_type": "Loan Estimate",
  "document_type_confidence": "high",
  "document_type_reason": "Rule-based (18 pts) and LLM agree: Contains Loan Estimate header, NMLS number, projected payments section",
  "fields": {
    "total_closing_costs": {
      "raw_value": "$11,475.00",
      "normalized_value": 11475.0,
      "source_page": 2,
      "confidence": "high",
      "confidence_reason": null,
      "evidence": "Correction on page 2 supersedes original $9,920.00 per addendum rule",
      "corrected": true,
      "correction_source_page": 2,
      "correction_rule_applied": "Addendum/correction pages override original values. Latest explicit date wins; if undated, higher page number wins.",
      "correction_date": "03/18/2026"
    },
    "borrower_name": {
      "raw_value": "SMITH, JOHN A",
      "normalized_value": "SMITH, JOHN A",
      "source_page": 1,
      "confidence": "high",
      "confidence_reason": null,
      "evidence": "Borrower name appears at top of page 1 in standard TRID header block",
      "corrected": false
    }
  },
  "tables": [
    {
      "name": "Closing Cost Summary",
      "source_page": 1,
      "rows": [
        {"label": "Origination Charges", "amount": "$4,860.00", "notes": ""},
        {"label": "Services Borrower Did Not Shop For", "amount": "$1,240.00", "notes": ""}
      ],
      "confidence": "high"
    }
  ],
  "detected_elements": [
    {
      "element_type": "signature",
      "page": 3,
      "description": "Borrower signature line — blank placeholder, no ink signature present",
      "confidence": "high"
    }
  ],
  "degraded_text_blocks": [
    {
      "page": 3,
      "location": "bottom-right corner",
      "partial_text": "Borower Signature: ___ Date: 03/14/2O26",
      "reason": "Low-contrast scan with OCR substitution errors (O for 0)",
      "confidence": "low"
    }
  ],
  "corrections_applied": [
    {
      "field": "taxes_and_govt_fees",
      "original_value": "$9,920.00",
      "original_page": 1,
      "corrected_value": "$11,475.00",
      "correction_page": 2,
      "correction_date": "03/18/2026",
      "resolution_rule": "Addendum/correction pages override original values. Among multiple corrections to the same field: latest explicit date wins; if undated, higher page number wins."
    }
  ],
  "mismatches": [
    {
      "field": "total_closing_costs",
      "computed_value": "9920.0",
      "stated_value": "11475.0",
      "description": "D+I = total_closing_costs: stated total_closing_costs=11475.00, components sum to 9920.00 (diff $1555.00). Values preserved as-stated per pipeline policy.",
      "severity": "error",
      "action": "flagged"
    }
  ],
  "processing_time_seconds": 22.1,
  "pages_processed": 3
}

Architecture
mortgage-pipeline/
├── app.py                        # Streamlit UI — upload, progress, results tabs, JSON export
├── requirements.txt
├── .env                          # GROQ_API_KEY goes here (not committed)
├── fixtures/
│   ├── loan_doc.pdf              # Sample Loan Estimate
│   └── appraisal_doc.pdf         # Sample Appraisal Summary
└── pipeline/
    ├── __init__.py               # Package exports
    ├── models.py                 # Pydantic v2 data models (ExtractionResult, FieldValue, ...)
    ├── pdf_processor.py          # PyMuPDF page rendering + pdfplumber table extraction
    ├── vision_llm.py             # Groq vision calls, diskcache, exponential backoff
    ├── document_classifier.py    # Hybrid rule-based + LLM classification
    ├── correction_resolver.py    # 4-layer field resolution + correction rule engine
    ├── mismatch_detector.py      # Arithmetic total verification
    ├── text_cleaner.py           # OCR artifact normalization, monetary/% parsing
    ├── confidence_engine.py      # Page-quality-aware confidence adjustment
    └── orchestrator.py           # ThreadPoolExecutor pipeline coordinator
├── part2/
│   ├── design_doc.md          ← Part 2 design document
│   ├── README_part2.md
│   └── label_studio/
│       ├── setup_windows.bat
│       ├── labeling_config.xml
│       ├── export_to_ls.py
│       └── import_corrections.py
└── screenshots/               
    ├── loan_doc_fields.png
    ├── loan_doc_corrections.png
    └── label_studio_dryrun.png

How the pipeline satisfies each requirement
Requirement 1 — Every field has source_page and confidence
Every FieldValue object in the output carries:

source_page — the page number the LLM read the value from
confidence — high / medium / low
confidence_reason — one-line explanation when not high
evidence — the exact text excerpt from the document
The confidence_engine.py module post-processes all resolved fields and can downgrade confidence when:

The value fails format validation (e.g. a monetary field contains non-numeric text)
The source page was flagged as degraded by the vision model
Requirement 2 — Explicit, statable correction rule
The rule:

A page must be explicitly identified as an ADDENDUM, CORRECTION, REVISION, or AMENDMENT (by keyword in the page header or via the vision LLM's is_correction_page flag) to override an earlier figure. Among multiple corrections to the same field: the one with the latest explicit date wins. If no correction carries an explicit date, the higher page number wins.

This is enforced in correction_resolver.py. The rule is:

Not a hardcoded "last page wins" assumption
Not a "highest page number always wins" rule
A general rule that requires explicit labeling as a correction page
Correction records include correction_date, original_page, correction_page, and resolution_rule so the decision is fully auditable.

Field name resolution uses a 4-layer approach so that LLM-generated field names (which may vary in phrasing) reliably map to canonical schema names:

Exact alias match — a hand-coded synonym dictionary (fastest)
Exact schema name match
Semantic similarity via sentence-transformers all-MiniLM-L6-v2 — understands meaning, not just spelling (e.g. "county recorder fee" → taxes_and_govt_fees)
rapidfuzz string similarity — last-resort fuzzy fallback
Requirement 3 — Deliberate mismatch policy
Decision: flag and preserve as-stated. Never recompute.

Reasoning: the source document is the legal record of truth. When components A + B + C do not sum to the stated total D (after applying corrections), this is a real discrepancy in the document — potentially a tolerance cure, a rounding difference, or a genuine error. Silently rewriting D would hide that discrepancy from the reviewer and create a pipeline output that disagrees with the original document without any audit trail.

Implemented in mismatch_detector.py:

Checks total_loan_costs = A + B + C (origination + services not shopped + services shopped)
Checks total_other_costs = E + F + G + H (taxes + prepaids + escrow + other)
Checks total_closing_costs = D + I (total loan costs + total other costs)
Checks LTV identity: loan_amount / appraised_value vs stated LTV
Generic pass: finds any total_* field and tests whether a plausible subset of numeric fields sums to it
Each MismatchFlag carries stated_value, computed_value, difference, severity (error / warning), and action: "flagged".

Requirement 4 — Same schema for all document types
ExtractionResult and FieldValue are identical regardless of whether the input is a Loan Estimate, Appraisal, or Closing Disclosure. The vision LLM prompt is document-type agnostic — it instructs the model to extract every field it can read on the page, using a comprehensive hint catalog covering all six mortgage document types as naming guidance only. Fields not present on a given page are simply absent; no doc-type-specific code paths exist.

Key design decisions
1. Groq Llama-4-Scout as vision + extraction in one pass

Each page is rendered to a 150–200 DPI image and sent to the Groq vision model alongside the raw text layer. The model handles noisy scans, rotated stamps, and faint watermarks better than open-source OCR alone. Adaptive DPI: sparse or blank text-layer pages are rendered at 200 DPI; native-text pages at 150 DPI.

2. Document-type-agnostic prompt

The vision prompt does not list a document-specific field set. It provides a canonical hint catalog (160+ field names across all mortgage document types) as naming guidance, then explicitly tells the LLM: "extract every field you can read, including fields not in this catalog." This is what pushed extraction from ~9 fields (hardcoded LE-only list) to 23–28 fields per document.

3. Disk cache keyed on page-image hash

Results are cached in .llm_cache/ with key v4:<sha256_of_image_bytes>. Re-running the same document costs $0 in API calls. Cache key is versioned so changing the prompt version-bumps the key and invalidates old results automatically.

4. Field merge: highest-confidence, first-page-preferred

When the same field appears on multiple non-correction pages, the pipeline keeps the highest-confidence value and breaks ties in favor of the lowest page number. This preserves borrower/loan identification fields from page 1 even when page 2 also mentions them. The correction resolver then overrides any field where a correction page supersedes the original.

5. Concurrency architecture

Pages within a document: ThreadPoolExecutor with configurable workers (1–5)
Groq API calls: threading.Semaphore(3) caps simultaneous calls across all workers
Multiple documents: Streamlit callers loop themselves (avoids NoSessionContext errors from nested threads)
Tradeoffs (cost / latency / accuracy)
Approach	Cost/page	Latency/page	Accuracy on scanned docs
Groq Llama-4-Scout (this pipeline)	~$0.00005	~2–4s	High
Groq Llama-3.2-11b-vision (fallback)	~$0.00002	~1.5–3s	Medium-High
GPT-4o-mini	~$0.0003	~2–4s	High
Gemini 1.5 Flash	~$0.00015	~1.5–3s	High
Tesseract + rules	$0	~0.2s	Medium (fails on noisy scans)
Surya OCR + rules	$0	~1s	Medium-High (no semantic understanding)
For a daily batch of 50 documents × 3 pages = 150 pages: **$0.008/day with Llama-4-Scout**.

Honest limitations (what I'd fix with more time)
Field recall on dense closing-cost tables. The LLM sometimes misses rows in large multi-column tables. A second pdfplumber pass that reconciles every table row against the extracted fields would close this gap.

LTV post-correction recalculation. The pipeline flags an LTV mismatch when loan_amount / appraised_value doesn't match the stated LTV, but doesn't automatically re-derive LTV from a corrected appraised value. A post-correction step would make this explicit rather than leaving it as a mismatch for the reviewer.

Multi-document correction chains. Corrections that span separate PDFs (a stand-alone addendum file) are not yet linked. The current rule only applies within a single PDF.

Exponential backoff jitter on rate limits. The Semaphore prevents bursts, but retry delay is linear. True exponential backoff with jitter (as in AWS SDK retry logic) would be more robust under sustained high volume.

Signature detection accuracy. The pipeline correctly flags blank signature lines as borrower_signature_present: false. Distinguishing a very faint ink signature from a blank underline is still unreliable at 150 DPI; bumping to 300 DPI on signature pages would help.