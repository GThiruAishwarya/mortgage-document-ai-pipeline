# Part 2 — Operability Design

## Overview

Part 1 produces structured JSON from mortgage PDFs. Some fields come back with `confidence: "low"` — values the pipeline is uncertain about. A non-technical mortgage ops analyst needs to review those fields, correct them where wrong, and have those corrections actually improve the system over time.

This document covers four things: the human review tool, the CI/CD feedback loop, what "retraining" means in this context, and a plain-English explanation for the analyst.

---

## 1. Human Review Tool — Label Studio (self-hosted)

### Why Label Studio

Label Studio is open-source, runs entirely locally (no cloud account needed), has a REST API for programmatic import/export, and supports custom HTML/JSON labeling interfaces. For mortgage ops, the alternative would be a custom-built web form — Label Studio gives us 80% of that for free.

### Local setup

See `label_studio/setup_windows.bat` (Windows) or `label_studio/setup_unix.sh` (Linux/macOS) for one-command setup. Manual steps:

```bash
# 1. Install Label Studio
pip install label-studio

# 2. Start the server (runs at http://localhost:8080)
label-studio start

# 3. Sign up with any email/password (local only — no account verification)
# 4. Create a new project called "Mortgage Review"
# 5. Use the labeling config from label_studio/labeling_config.xml
```

### Loading low-confidence extractions

After running the Part 1 pipeline and exporting JSON, run:

```bash
python label_studio/export_to_ls.py \
  --extraction output/loan_doc_result.json \
  --project-id 1 \
  --label-studio-url http://localhost:8080 \
  --api-token YOUR_LS_TOKEN
```

This script:
1. Reads the `ExtractionResult` JSON
2. Filters to fields where `confidence == "low"` (and optionally `"medium"`)
3. Formats each field as a Label Studio task showing: field name, current extracted value, source page, confidence reason, and evidence text
4. POSTs the tasks to Label Studio via its REST API

### What the reviewer sees

Each task shows one low-confidence field at a time:

```
Field:         taxes_and_govt_fees
Current value: $4,320.00
Source page:   1
Reason:        Value on correction page conflicts with page 1; date unclear
Evidence:      "Section E. Taxes and Govt Fees  $4,320.00"

┌─────────────────────────────────────┐
│ Corrected value: [______________]   │
│ Confidence:  ○ Correct  ○ Wrong  ○ Unsure │
│ Notes:       [______________]       │
└─────────────────────────────────────┘
```

The reviewer types the correct value (or confirms the extracted value is right), picks a confidence verdict, and optionally adds a note. They do not need to touch any code or JSON.

### Importing corrections back

```bash
python label_studio/import_corrections.py \
  --project-id 1 \
  --label-studio-url http://localhost:8080 \
  --api-token YOUR_LS_TOKEN \
  --output corrections/loan_doc_corrections_2026-06-21.json
```

This produces a corrections file like:

```json
[
  {
    "document": "loan_doc.pdf",
    "field": "taxes_and_govt_fees",
    "original_pipeline_value": "$4,320.00",
    "reviewer_corrected_value": "$5,890.00",
    "reviewer_verdict": "wrong",
    "notes": "Correction addendum dated 04/15 clearly shows $5,890",
    "reviewed_at": "2026-06-21T14:32:00Z",
    "reviewer": "analyst@finacplus.com"
  }
]
```

This corrections file feeds the CI/CD pipeline described in Section 2.

### Labeling config

The full XML interface is in `label_studio/labeling_config.xml`. It uses Label Studio's `<Text>` and `<TextArea>` components — no custom code required.

---

## 2. CI/CD Plan

### The story for a non-technical stakeholder

> A reviewer corrects a field in the review tool. That correction gets saved to a file. A nightly job (Jenkins) picks up that file, runs it through automated checks, and if the checks pass, schedules the improvement for the next day's pipeline run. A second human (the team lead) approves before it reaches production documents. Nothing ships automatically to production without a sign-off.

### Detailed flow (Jenkins / Forgejo terms)

```
Reviewer submits correction in Label Studio
          │
          ▼
[corrections/YYYY-MM-DD.json committed to Forgejo repo]
          │
          ▼
Jenkins detects new commit → triggers "Correction Review" pipeline
          │
          ├─ Stage 1: Validate correction format
          │   • JSON schema check (all required fields present)
          │   • Monetary format check ($-prefixed, parseable float)
          │   • Duplicate detection (same field already corrected this week?)
          │   GATE: fails here if file is malformed → engineer notified, not ops
          │
          ├─ Stage 2: Regression test
          │   • Run pipeline on the 10 most recent documents (from fixtures/)
          │   • Compare output to known-good baseline snapshots
          │   • Flag if any previously HIGH-confidence field changed value
          │   GATE: any regression → pipeline stops, engineer reviews
          │
          ├─ Stage 3: Prompt/alias update (automated)
          │   • If correction is a new field alias (e.g. "county_tax" → "taxes_and_govt_fees"),
          │     add it to correction_resolver.py's _FIELD_ALIASES dict automatically
          │   • If correction is a few-shot example, append it to the prompt example bank
          │   • Commit the change to a feature branch in Forgejo
          │
          └─ Stage 4: Human approval gate (REQUIRED before production)
              • Jenkins posts a Forgejo pull request for the team lead to review
              • PR shows: what changed, what was corrected, regression test results
              • Team lead clicks "Merge" in Forgejo UI — no code knowledge required
              • After merge, Jenkins redeploys the pipeline with the new version
```

### What is automatic vs. human-gated

| Step | Automatic | Requires human |
|---|---|---|
| Format validation | ✅ | |
| Regression test | ✅ | |
| Field alias update | ✅ | |
| Prompt example append | ✅ | |
| Push to production | | ✅ Team lead merge |
| Schema model changes | | ✅ Engineer review |
| Threshold adjustments | | ✅ Engineer review |

### Forgejo-specific notes

- The correction file lives in a `corrections/` directory in the main repo
- Branch protection on `main` requires 1 reviewer approval before merge
- Jenkins webhook triggers on push to `corrections/` path only (not all pushes)
- A nightly Jenkins cron job also runs the full regression suite against the latest `main`

---

## 3. Definition of "Retraining"

### What "retraining" means here

This pipeline does not train a neural network — it calls a hosted LLM (Groq) via API. So "retraining" means **improving what we give the LLM to work with**, not fine-tuning model weights. Three levers exist, in order of effort:

**Lever 1 — Field alias expansion (lowest effort, highest frequency)**

When a reviewer corrects a field that was extracted under the wrong name (e.g. the LLM returned `"county_recorder_fee"` but the canonical name is `"taxes_and_govt_fees"`), we add the mapping to `_FIELD_ALIASES` in `correction_resolver.py`. This is a one-line change per correction, runs in CI automatically, and takes effect immediately on the next pipeline run.

*When:* every time a reviewer corrects a field name mapping. In practice, weekly in the first month, tapering to monthly once the alias dict is mature.

**Lever 2 — Few-shot example bank (medium effort, monthly)**

The vision LLM prompt includes a hint catalog of preferred field names. Over time, we add concrete examples drawn from reviewer corrections — pairs of `(field_name, example_raw_value)` — to anchor the LLM's extraction. This is a prompt file update, not a model fine-tune. It improves consistency for field types that keep coming back wrong.

*When:* when the same field is corrected 3+ times in a month — that's a signal the prompt is ambiguous for that field type, not that the document is unusual.

*Trigger threshold:* 3 corrections to the same field in 30 days → auto-open a Forgejo issue suggesting a few-shot addition.

**Lever 3 — Confidence threshold calibration (low frequency, engineering-led)**

We track the rate at which reviewers override `high`-confidence vs `medium`-confidence vs `low`-confidence fields. If we're seeing a 40% override rate on `medium`-confidence fields, the threshold that separates `medium` from `high` is too permissive and should be tightened. This is a change to `confidence_engine.py` thresholds.

*When:* quarterly review, or when override rate on any confidence tier exceeds 20%.

### What we explicitly do NOT do

- We do not fine-tune the Groq model (not supported on hosted API; not worth the cost at this volume)
- We do not "retrain" based on individual corrections — one correction is anecdote, three is signal, ten is pattern
- We do not automatically change the prompt without a passing regression test and team lead approval

---

## 4. Plain-English Explanation for the Analyst

Every day, the pipeline reads the mortgage documents and pulls out the important numbers — loan amounts, costs, borrower names, and so on. When it's not sure about a value, it flags it in yellow or red instead of guessing quietly.

When you see a flag, you open the review tool (it runs right on your computer — no internet needed), see the value the pipeline found alongside the actual text from the document, and either confirm it's right or type in the correct number. That's it — you never touch a spreadsheet or a config file.

Once a week, those corrections are automatically checked and, after a quick sign-off from the team lead, get folded back into the pipeline so the same mistake is less likely to happen again. Over time the pipeline gets more accurate on the kinds of documents you process most often.

If something goes wrong — the pipeline crashes, produces a nonsense value, or misses a whole section — the engineer gets notified automatically and nothing ships to production until it's fixed.

---

## File listing

```
part2/
├── design_doc.md                     ← this file
├── README_part2.md                   ← setup and run instructions
└── label_studio/
    ├── setup_windows.bat             ← one-command Windows setup
    ├── setup_unix.sh                 ← one-command Linux/macOS setup
    ├── labeling_config.xml           ← Label Studio interface definition
    ├── export_to_ls.py               ← export Part 1 JSON → Label Studio tasks
    └── import_corrections.py         ← export Label Studio annotations → corrections JSON
```
