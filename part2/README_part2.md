# Part 2 — Operability Design: Setup & Run Guide


## What's in this folder

| File | Purpose |
|---|---|
| `design_doc.md` | Full Part 2 design document (human review, CI/CD, retraining, operator explanation) |
| `label_studio/setup_windows.bat` | One-command Label Studio setup for Windows |
| `label_studio/setup_unix.sh` | One-command Label Studio setup for Linux / macOS |
| `label_studio/labeling_config.xml` | Label Studio interface definition for mortgage field review |
| `label_studio/export_to_ls.py` | Exports Part 1 extraction JSON → Label Studio tasks |
| `label_studio/import_corrections.py` | Exports Label Studio annotations → corrections JSON file |

---

## Step-by-step: running Label Studio locally

### Step 1 — Install Label Studio

**Windows:**
```cmd
setup_windows.bat
```

**Linux / macOS:**
```bash
chmod +x label_studio/setup_unix.sh
./label_studio/setup_unix.sh
```

**Or manually (any platform):**
```bash
pip install label-studio
```

Label Studio requires Python 3.8+. It has no other system dependencies.

---

### Step 2 — Start the Label Studio server

```bash
label-studio start
```

This launches a local web server. Open **http://localhost:8080** in your browser.

- First time: click "Create Account" — use any email and password. This is local only; no email is sent.
- You will land on the Label Studio projects page.

---

### Step 3 — Create a project

1. Click **Create Project**
2. Name it: `Mortgage Field Review`
3. Skip the data import for now (we will import via script)
4. Go to the **Labeling Setup** tab
5. Click **Code** (switch to raw XML mode)
6. Paste the contents of `label_studio/labeling_config.xml`
7. Click **Save**

---

### Step 4 — Get your API token

1. Click your avatar (top-right) → **Account & Settings**
2. Copy the **Access Token** shown on that page
3. You will pass this as `--api-token` to the scripts below

---

### Step 5 — Export a Part 1 result into Label Studio

First, run the Part 1 pipeline (from the `mortgage-pipeline/` folder) on one of the fixture PDFs and export the result as JSON from the UI's **Export JSON** tab. Save it as e.g. `output/loan_doc_result.json`.

Then:

```bash
python label_studio/export_to_ls.py \
  --extraction output/loan_doc_result.json \
  --project-id 1 \
  --label-studio-url http://localhost:8080 \
  --api-token YOUR_TOKEN_HERE
```

Options:
- `--min-confidence low` — only import `low` confidence fields (default)
- `--min-confidence medium` — import both `medium` and `low` confidence fields
- `--include-all` — import every field regardless of confidence

---

### Step 6 — Review in Label Studio

1. Open **http://localhost:8080** → click your project
2. Click **Label** (the blue button)
3. For each task you will see:
   - The field name, current pipeline value, source page, and evidence text
   - A text box to type the corrected value (or leave it to confirm the current value)
   - A confidence verdict: **Correct** / **Wrong** / **Unsure**
   - An optional notes field
4. Click **Submit** to save and move to the next field
5. When done, click **Tasks** tab to see completion status

---

### Step 7 — Export corrections back to a JSON file

```bash
python label_studio/import_corrections.py \
  --project-id 1 \
  --label-studio-url http://localhost:8080 \
  --api-token YOUR_TOKEN_HERE \
  --output corrections/loan_doc_corrections.json
```

This produces a corrections JSON file (see `design_doc.md` for the schema). In a full deployment, this file would be committed to the repo and picked up by the Jenkins CI/CD pipeline.

---

## Common issues

### "label-studio: command not found" after installing

The `label-studio` binary is in your Python `Scripts/` folder. Make sure it is on your PATH:

```bash
# Check where it was installed
pip show label-studio | grep Location

# Add to PATH (Linux/macOS — add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Windows: add the Scripts folder to System PATH via Control Panel
```

### Port 8080 already in use

```bash
label-studio start --port 8090
```

Then use `http://localhost:8090` and pass `--label-studio-url http://localhost:8090` to the scripts.

### Script errors: "requests not installed"

```bash
pip install requests
```

### Label Studio asks for a Django secret key

Set it before starting:
```bash
# Linux/macOS
export LABEL_STUDIO_SECRET_KEY=any-random-string-here

# Windows
set LABEL_STUDIO_SECRET_KEY=any-random-string-here
```

---

## Running without Label Studio (quick demo mode)

If you just want to see what low-confidence fields look like without setting up Label Studio:

```bash
python label_studio/export_to_ls.py \
  --extraction output/loan_doc_result.json \
  --dry-run
```

This prints the tasks to the terminal instead of uploading them.
