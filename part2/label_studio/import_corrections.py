"""
import_corrections.py
=====================
Exports completed Label Studio annotations back into a structured
corrections JSON file that can be fed into the CI/CD pipeline.

Usage:
    python import_corrections.py \\
        --project-id 1 \\
        --label-studio-url http://localhost:8080 \\
        --api-token YOUR_TOKEN \\
        --output corrections/loan_doc_corrections.json

Output schema (one object per corrected field):
    {
        "document": "loan_doc.pdf",
        "field": "taxes_and_govt_fees",
        "field_display": "Taxes And Govt Fees",
        "original_pipeline_value": "$4,320.00",
        "reviewer_verdict": "wrong",          // correct | wrong | unsure
        "reviewer_corrected_value": "$5,890.00",
        "reviewer_notes": "Correction addendum page 2 shows $5,890",
        "source_page": "Page 1",
        "reviewed_at": "2026-06-21T14:32:00Z"
    }
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed. Run: pip install requests")
    sys.exit(1)


def fetch_annotations(project_id: int, base_url: str, token: str) -> list:
    """Fetch all completed tasks (with annotations) from the Label Studio project."""
    url = f"{base_url.rstrip('/')}/api/projects/{project_id}/export"
    headers = {"Authorization": f"Token {token}"}
    params = {"exportType": "JSON"}

    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"ERROR fetching annotations: HTTP {response.status_code}")
        print(response.text[:500])
        sys.exit(1)


def parse_annotation(task: dict) -> dict | None:
    """
    Extract the reviewer's answers from a Label Studio task annotation.

    Returns None if the task has no completed annotations.
    """
    annotations = task.get("annotations", [])
    if not annotations:
        return None

    latest = annotations[-1]
    results = latest.get("result", [])

    verdict = None
    corrected_value = ""
    reviewer_notes = ""

    for r in results:
        name = r.get("from_name", "")
        value = r.get("value", {})

        if name == "verdict":
            choices = value.get("choices", [])
            if choices:
                verdict = choices[0]

        elif name == "corrected_value":
            texts = value.get("text", [])
            if texts:
                corrected_value = texts[0].strip()

        elif name == "reviewer_notes":
            texts = value.get("text", [])
            if texts:
                reviewer_notes = texts[0].strip()

    if verdict is None:
        return None

    data = task.get("data", {})

    completed_at = latest.get("completed_at") or datetime.now(timezone.utc).isoformat()

    return {
        "document": data.get("source_document", "unknown"),
        "field": data.get("field_key", ""),
        "field_display": data.get("field_name", ""),
        "original_pipeline_value": data.get("extracted_value", ""),
        "source_page": data.get("source_page", ""),
        "reviewer_verdict": verdict,
        "reviewer_corrected_value": corrected_value,
        "reviewer_notes": reviewer_notes,
        "reviewed_at": completed_at,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export Label Studio annotations to a corrections JSON file"
    )
    parser.add_argument(
        "--project-id", type=int, default=1,
        help="Label Studio project ID"
    )
    parser.add_argument(
        "--label-studio-url", default="http://localhost:8080",
        help="Label Studio base URL"
    )
    parser.add_argument(
        "--api-token", required=True,
        help="Label Studio API token"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for corrections JSON (e.g. corrections/loan_doc_corrections.json)"
    )
    parser.add_argument(
        "--only-wrong", action="store_true",
        help="Only include fields where reviewer verdict was 'wrong' (skip 'correct' and 'unsure')"
    )
    args = parser.parse_args()

    print(f"Fetching annotations from project {args.project_id}...")
    tasks = fetch_annotations(args.project_id, args.label_studio_url, args.api_token)
    print(f"Fetched {len(tasks)} task(s)")

    corrections = []
    skipped = 0

    for task in tasks:
        result = parse_annotation(task)
        if result is None:
            skipped += 1
            continue

        if args.only_wrong and result["reviewer_verdict"] != "wrong":
            skipped += 1
            continue

        corrections.append(result)

    print(f"Parsed {len(corrections)} annotation(s) ({skipped} skipped / incomplete)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)

    print(f"Corrections saved to: {output_path}")
    print()

    # Summary
    wrong_count = sum(1 for c in corrections if c["reviewer_verdict"] == "wrong")
    correct_count = sum(1 for c in corrections if c["reviewer_verdict"] == "correct")
    unsure_count = sum(1 for c in corrections if c["reviewer_verdict"] == "unsure")

    print(f"Summary:")
    print(f"  ✅ Confirmed correct: {correct_count}")
    print(f"  ❌ Corrected (wrong): {wrong_count}")
    print(f"  ⚠️  Flagged unsure:   {unsure_count}")

    if wrong_count > 0:
        print(f"\nFields marked wrong:")
        for c in corrections:
            if c["reviewer_verdict"] == "wrong":
                print(
                    f"  {c['field']}: '{c['original_pipeline_value']}' → '{c['reviewer_corrected_value']}'"
                )


if __name__ == "__main__":
    main()
