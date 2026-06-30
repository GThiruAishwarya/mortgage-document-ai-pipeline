"""
export_to_ls.py
================
Exports fields from a Part 1 ExtractionResult JSON into Label Studio
tasks via the Label Studio REST API.

Authentication — pick ONE of:
  --api-token YOUR_TOKEN          (from Personal Access Token page)
  --username EMAIL --password PW  (logs in automatically, no token needed)

Usage examples:
    # With username/password (easiest):
    python export_to_ls.py ^
        --extraction loan_doc1_extraction.json ^
        --username you@email.com --password yourpass ^
        --label-studio-url http://localhost:8081 ^
        --min-confidence all

    # Dry-run (prints tasks, no upload):
    python export_to_ls.py ^
        --extraction loan_doc1_extraction.json ^
        --username you@email.com --password yourpass ^
        --label-studio-url http://localhost:8081 ^
        --min-confidence all --dry-run

    # With token:
    python export_to_ls.py ^
        --extraction loan_doc1_extraction.json ^
        --api-token YOUR_TOKEN ^
        --label-studio-url http://localhost:8081 ^
        --min-confidence all

Options:
    --min-confidence   low|medium|all  (default: low)
    --dry-run          Print tasks to terminal instead of uploading
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed. Run: pip install requests")
    sys.exit(1)


CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_token_from_credentials(base_url: str, username: str, password: str) -> str:
    """
    Log in with email + password and return the API token.
    Works with Label Studio 1.x and 2.x.
    """
    url = f"{base_url.rstrip('/')}/api/token"
    print(f"Logging in as {username} ...")

    resp = requests.post(
        url,
        json={"username": username, "password": password},
        timeout=15,
    )

    if resp.status_code == 200:
        data = resp.json()
        token = data.get("token") or data.get("access") or data.get("key")
        if token:
            print("Login successful — token acquired.")
            return token
        print(f"ERROR: Login response did not contain a token. Response: {data}")
        sys.exit(1)

    # Some Label Studio versions use /auth/login (session-based) + /api/current-user/token
    if resp.status_code in (400, 404, 405):
        print("Trying alternative login endpoint (/auth/login) ...")
        session = requests.Session()
        # Step 1: get CSRF token
        login_page = session.get(f"{base_url.rstrip('/')}/user/login/", timeout=10)
        csrf = ""
        for line in login_page.text.splitlines():
            if "csrfmiddlewaretoken" in line:
                start = line.find('value="') + 7
                end = line.find('"', start)
                csrf = line[start:end]
                break

        # Step 2: POST login form
        login_resp = session.post(
            f"{base_url.rstrip('/')}/user/login/",
            data={
                "csrfmiddlewaretoken": csrf,
                "email": username,
                "password": password,
            },
            headers={"Referer": f"{base_url.rstrip('/')}/user/login/"},
            timeout=15,
        )

        if login_resp.status_code not in (200, 302):
            print(f"ERROR: Login failed (HTTP {login_resp.status_code})")
            print("Check your email and password.")
            sys.exit(1)

        # Step 3: fetch API token for this session
        token_resp = session.get(
            f"{base_url.rstrip('/')}/api/current-user/token",
            timeout=10,
        )
        if token_resp.status_code == 200:
            token = token_resp.json().get("token")
            if token:
                print("Login successful — token acquired via session.")
                return token

        print(f"ERROR: Could not retrieve token after login. HTTP {token_resp.status_code}")
        print(token_resp.text[:300])
        sys.exit(1)

    print(f"ERROR: Login failed (HTTP {resp.status_code}): {resp.text[:300]}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_extraction(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tasks(extraction: dict, min_confidence: str) -> list:
    """
    Convert ExtractionResult fields into Label Studio task dicts.

    min_confidence controls which fields are included:
      "high"   → all fields (high + medium + low)
      "medium" → medium + low confidence fields only
      "low"    → low confidence fields only
    """
    tasks = []
    filename = extraction.get("filename", "unknown.pdf")
    doc_type = extraction.get("document_type", "Unknown")
    min_rank = CONFIDENCE_RANK.get(min_confidence, 0)

    fields = extraction.get("fields", {})
    if not fields:
        print("WARNING: No 'fields' key found in the extraction JSON.")
        print(f"Top-level keys present: {list(extraction.keys())}")
        return []

    for field_name, fv in fields.items():
        conf = fv.get("confidence", "low")
        conf_rank = CONFIDENCE_RANK.get(conf, 2)

        # Skip fields whose confidence rank is BETTER than the minimum.
        # e.g. min_confidence="medium" (rank 1) → skip high (rank 0).
        if conf_rank < min_rank:
            continue

        doc_info = (
            f"Document: {filename}  |  Type: {doc_type}  |  "
            f"Confidence: {conf.upper()}"
        )

        display_name = field_name.replace("_", " ").title()
        page = fv.get("source_page", "?")
        raw_value = fv.get("raw_value", "")
        reason = fv.get("confidence_reason") or "High confidence — no issues flagged"
        evidence = fv.get("evidence") or "No evidence text captured"

        if len(evidence) > 400:
            evidence = evidence[:400] + "…"

        tasks.append({
            "data": {
                "doc_info": doc_info,
                "field_name": display_name,
                "field_key": field_name,
                "source_page": f"Page {page}",
                "extracted_value": raw_value,
                "confidence_reason": reason,
                "evidence": evidence,
                "source_document": filename,
            }
        })

    return tasks


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_tasks(tasks: list, project_id: int, base_url: str, token: str) -> None:
    url = f"{base_url.rstrip('/')}/api/projects/{project_id}/import"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }
    print(f"Uploading {len(tasks)} task(s) to project {project_id} ...")
    response = requests.post(url, json=tasks, headers=headers, timeout=30)

    if response.status_code in (200, 201):
        data = response.json()
        count = data.get("task_count", len(tasks))
        print(f"\nSUCCESS: {count} task(s) uploaded!")
        print(f"Open in browser: {base_url}/projects/{project_id}/data")
        print(f"Click the blue 'Label' button to start reviewing.")
    else:
        print(f"ERROR uploading tasks: HTTP {response.status_code}")
        print(response.text[:500])
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export Part 1 extraction JSON to Label Studio tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extraction", required=True,
        help="Path to ExtractionResult JSON file"
    )
    parser.add_argument(
        "--project-id", type=int, default=1,
        help="Label Studio project ID (default: 1)"
    )
    parser.add_argument(
        "--label-studio-url", default="http://localhost:8080",
        help="Label Studio base URL (default: http://localhost:8080)"
    )

    auth_group = parser.add_argument_group("Authentication (use token OR username+password)")
    auth_group.add_argument(
        "--api-token", default="",
        help="Label Studio personal access token"
    )
    auth_group.add_argument(
        "--username", default="",
        help="Label Studio login email"
    )
    auth_group.add_argument(
        "--password", default="",
        help="Label Studio login password"
    )

    parser.add_argument(
        "--min-confidence",
        choices=["low", "medium", "all"],
        default="low",
        help="Which fields to include: low=only low-conf, medium=medium+low, all=every field (default: low)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print tasks to terminal instead of uploading"
    )
    args = parser.parse_args()

    # --- Load extraction JSON ---
    extraction_path = Path(args.extraction)
    if not extraction_path.exists():
        print(f"ERROR: File not found: {args.extraction}")
        sys.exit(1)

    print(f"Loading extraction: {args.extraction}")
    extraction = load_extraction(str(extraction_path))
    print(f"Document: {extraction.get('filename', '?')} — {extraction.get('document_type', '?')}")
    print(f"Fields in JSON: {len(extraction.get('fields', {}))}")

    # --- Build tasks ---
    # "all" → include everything (rank >= 0 = "high")
    # "medium" → include medium + low (rank >= 1)
    # "low" → include only low (rank >= 2)
    confidence_map = {"all": "high", "medium": "medium", "low": "low"}
    min_conf = confidence_map[args.min_confidence]
    tasks = build_tasks(extraction, min_conf)

    if not tasks:
        print(f"\nNo fields matched --min-confidence '{args.min_confidence}'.")
        if args.min_confidence == "low":
            print("All fields are HIGH or MEDIUM confidence — try: --min-confidence all")
        sys.exit(0)

    print(f"Fields to upload: {len(tasks)}  (filter: --min-confidence {args.min_confidence})\n")

    # --- Dry run ---
    if args.dry_run:
        print("--- DRY RUN (no upload) ---\n")
        for i, task in enumerate(tasks, 1):
            d = task["data"]
            print(f"  [{i:2d}] {d['field_name']}")
            print(f"        Value    : {d['extracted_value']}")
            print(f"        Page     : {d['source_page']}")
            print(f"        Confidence reason: {d['confidence_reason']}")
            print()
        print(f"Total: {len(tasks)} task(s) — remove --dry-run to upload.")
        return

    # --- Resolve token ---
    token = args.api_token.strip()

    if not token:
        if args.username and args.password:
            token = get_token_from_credentials(
                args.label_studio_url, args.username, args.password
            )
        else:
            print("ERROR: Provide either --api-token OR both --username and --password.")
            sys.exit(1)

    # --- Upload ---
    upload_tasks(tasks, args.project_id, args.label_studio_url, token)


if __name__ == "__main__":
    main()
