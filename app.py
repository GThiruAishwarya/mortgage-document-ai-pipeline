import os
import sys
import json
import tempfile
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pipeline import DocumentOrchestrator
from pipeline.models import ExtractionResult

st.set_page_config(
    page_title="Mortgage Document AI Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

CONFIDENCE_COLORS = {
    "high": "🟢",
    "medium": "🟡",
    "low": "🔴",
}

CONFIDENCE_LABELS = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}


def confidence_badge(conf: str) -> str:
    return f"{CONFIDENCE_COLORS.get(conf, '⚪')} {CONFIDENCE_LABELS.get(conf, conf.upper())}"


def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Pipeline Settings")
    st.sidebar.markdown("---")

    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=GROQ_API_KEY,
        type="password",
        help="Uses GROQ_API_KEY env var if set. Get a free key at https://console.groq.com",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")
    model_options = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
    ]
    model_choice = st.sidebar.selectbox(
        "Groq vision model",
        model_options,
        index=0,
        help="llama-4-scout is the recommended default. Falls back to llama-3.2-11b automatically if needed.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Concurrency")
    max_workers = st.sidebar.slider(
        "Page analysis workers",
        min_value=1, max_value=5, value=1,
        help="Concurrent Groq API calls per document. Keep at 1 if you hit 429 quota errors.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "**Pipeline Conditions:**\n"
        "✅ Multiple PDFs at once\n"
        "✅ Concurrent processing\n"
        "✅ Table / stamp / signature / watermark detection\n"
        "✅ Text cleaning (OCR artifacts)\n"
        "✅ Important field extraction\n"
        "✅ Context-aware classification\n"
        "✅ Confidence with evidence\n"
        "✅ Explicit correction resolution\n"
        "✅ Total mismatch detection"
    )

    return {"api_key": api_key, "max_workers": max_workers, "model": model_choice}


def render_fields_tab(result: ExtractionResult):
    st.subheader("Extracted Fields")

    if not result.fields:
        st.info("No fields extracted.")
        return

    rows = []
    for field_name, fv in result.fields.items():
        rows.append({
            "Field": field_name.replace("_", " ").title(),
            "Value": fv.raw_value,
            "Normalized": str(fv.normalized_value) if fv.normalized_value is not None else "",
            "Confidence": confidence_badge(fv.confidence),
            "Source Page": fv.source_page,
            "Corrected": "✅" if fv.corrected else "",
            "Reason": fv.confidence_reason or "",
            "Evidence": (fv.evidence[:80] + "…") if fv.evidence and len(fv.evidence) > 80 else (fv.evidence or ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    low_conf = [(fn, fv) for fn, fv in result.fields.items() if fv.confidence == "low"]
    if low_conf:
        st.warning(f"⚠️ {len(low_conf)} field(s) with LOW confidence need human review:")
        for fn, fv in low_conf:
            with st.expander(f"🔴 {fn.replace('_', ' ').title()} — page {fv.source_page}"):
                st.write(f"**Raw value:** `{fv.raw_value}`")
                st.write(f"**Reason:** {fv.confidence_reason or 'Not specified'}")
                st.write(f"**Evidence:** {fv.evidence or 'None'}")


def render_tables_tab(result: ExtractionResult):
    st.subheader("Extracted Tables")

    if not result.tables:
        st.info("No tables extracted.")
        return

    for tbl in result.tables:
        with st.expander(f"📋 {tbl.name} — Page {tbl.source_page} [{confidence_badge(tbl.confidence)}]"):
            if tbl.rows:
                df = pd.DataFrame(tbl.rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.write("(empty table)")


def render_elements_tab(result: ExtractionResult):
    st.subheader("Detected Visual Elements")
    st.caption("Stamps, signatures, watermarks, and other non-text elements detected by vision analysis.")

    if not result.detected_elements:
        st.info("No visual elements detected.")
        return

    type_icons = {
        "signature": "✍️",
        "stamp": "📮",
        "watermark": "💧",
        "logo": "🏷️",
        "image": "🖼️",
        "handwriting": "🖊️",
        "checkbox": "☑️",
    }

    for elem in result.detected_elements:
        icon = type_icons.get(elem.element_type, "🔲")
        col1, col2, col3, col4 = st.columns([1, 2, 3, 1])
        col1.write(f"{icon} **{elem.element_type.upper()}**")
        col2.write(f"Page {elem.page}")
        col3.write(elem.description)
        col4.write(confidence_badge(elem.confidence))


def render_corrections_tab(result: ExtractionResult):
    st.subheader("Correction Resolution")

    if result.corrections_applied:
        st.success(f"✅ {len(result.corrections_applied)} correction(s) applied using explicit rule.")
        with st.expander("📜 Correction Resolution Rule", expanded=False):
            st.info(
                "**Rule:** Any page explicitly identified as an ADDENDUM, CORRECTION, or REVISION "
                "supersedes the original figure on the page it references.\n\n"
                "**Tie-breaking:** Latest explicit date wins; if undated, higher page number wins.\n\n"
                "**This is not a 'last page wins' assumption** — the page must be explicitly labeled "
                "as a correction/addendum to trigger this rule."
            )

        for corr in result.corrections_applied:
            with st.expander(f"📝 {corr.field.replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)
                col1.metric("Original (Page {})".format(corr.original_page), corr.original_value)
                col2.metric(
                    "Corrected (Page {})".format(corr.correction_page),
                    corr.corrected_value,
                    delta="corrected",
                    delta_color="inverse",
                )
                if corr.correction_date:
                    st.write(f"**Correction date:** {corr.correction_date}")
                st.write(f"**Rule applied:** {corr.resolution_rule}")
    else:
        st.info("No corrections detected in this document.")


def render_mismatches_tab(result: ExtractionResult):
    st.subheader("Arithmetic Mismatch Detection")
    st.caption("Checks totals/subtotals against their components. Mismatches are flagged and preserved as-stated.")

    if not result.mismatches:
        st.success("✅ All verified totals reconcile correctly.")
        return

    for mm in result.mismatches:
        severity_icon = "🔴" if mm.severity == "error" else "🟡"
        with st.expander(f"{severity_icon} {mm.field.replace('_', ' ').title()} — {mm.severity.upper()}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Stated Value", mm.stated_value)
            if mm.computed_value:
                col2.metric("Computed Value", mm.computed_value)
            col3.write(f"**Action:** {mm.action.replace('_', ' ').title()}")
            st.write(f"**Description:** {mm.description}")


def render_degraded_blocks_tab(result: ExtractionResult):
    st.subheader("Degraded Text Blocks")
    st.caption(
        "Sections of the document where text quality was poor enough to affect extraction confidence. "
        "The pipeline records the best partial read, the location, and the cause."
    )

    if not result.degraded_text_blocks:
        st.success("✅ No significantly degraded text blocks detected.")
        return

    for blk in result.degraded_text_blocks:
        conf_icon = "🟡" if blk.confidence == "medium" else "🔴"
        with st.expander(f"{conf_icon} Page {blk.page} — {blk.location}"):
            col1, col2 = st.columns([2, 1])
            col1.write("**Partial text recovered:**")
            col1.code(blk.partial_text or "(none recoverable)")
            col2.write(f"**Cause:** {blk.reason}")
            col2.write(f"**Confidence:** {conf_icon} {blk.confidence.upper()}")


def render_page_quality_tab(result: ExtractionResult):
    st.subheader("Page Quality Analysis")

    if not result.page_analyses:
        st.info("No page data.")
        return

    quality_icons = {"clean": "🟢", "noisy": "🟡", "degraded": "🟠", "unreadable": "🔴"}
    rows = []
    for pa in result.page_analyses:
        rows.append({
            "Page": pa.page_number,
            "Quality": f"{quality_icons.get(pa.text_quality, '⚪')} {pa.text_quality.title()}",
            "Scanned": "Yes" if pa.is_scanned else "No",
            "Degraded Text": "Yes" if pa.has_degraded_text else "No",
            "Correction Page": "✅" if pa.is_correction_page else "",
            "Text Length": len(pa.raw_text),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_result(result: ExtractionResult, doc_idx: int):
    conf_icon = CONFIDENCE_COLORS.get(result.document_type_confidence, "⚪")

    if result.pages_processed == 0:
        st.error(f"❌ Processing failed for **{result.filename}**")
        reason = result.document_type_reason or "Unknown error (no details captured)"
        st.code(reason, language="text")
        reason_lower = reason.lower()
        if "429" in reason or "quota" in reason_lower or "rate_limit" in reason_lower:
            st.warning(
                "**Rate limit hit (429)**\n\n"
                "Your Groq API key hit its per-minute limit.\n\n"
                "**Try these steps:**\n"
                "1. Wait 30–60 seconds and click Run Pipeline again\n"
                "2. Set **Page analysis workers to 1** in the sidebar\n"
                "3. Switch to **llama-3.2-11b-vision-preview** (lower rate class)"
            )
        elif "404" in reason or "not_found" in reason_lower:
            st.warning(
                "**Model not found (404)**\n\n"
                "Switch to **llama-3.2-11b-vision-preview** in the sidebar."
            )
        elif "503" in reason or "unavailable" in reason_lower:
            st.warning("**Service temporarily unavailable (503)** — wait a minute and try again.")
        elif "api_key" in reason_lower or "invalid" in reason_lower or "401" in reason or "403" in reason:
            st.warning(
                "**Invalid API key (401/403)**\n\n"
                "Go to https://console.groq.com, copy your key, and paste it in the sidebar."
            )
        else:
            st.info("Common causes: invalid/expired API key, quota exceeded, network error.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Document Type", result.document_type)
    col2.metric("Classification Confidence", f"{conf_icon} {result.document_type_confidence.upper()}")
    col3.metric("Pages Processed", result.pages_processed)
    col4.metric("Processing Time", f"{result.processing_time_seconds}s")

    st.caption(f"**Classification reason:** {result.document_type_reason}")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Fields Extracted", len(result.fields))
    summary_cols[1].metric("Tables Found", len(result.tables))
    summary_cols[2].metric("Corrections Applied", len(result.corrections_applied))
    summary_cols[3].metric("Mismatches Flagged", len(result.mismatches))

    if result.mismatches:
        st.warning(f"⚠️ {len(result.mismatches)} arithmetic mismatch(es) detected — see Mismatches tab.")

    tabs = st.tabs([
        "📊 Fields",
        "📋 Tables",
        "🔍 Visual Elements",
        "🧩 Degraded Blocks",
        "📝 Corrections",
        "⚠️ Mismatches",
        "🖥️ Page Quality",
        "📥 Export JSON",
    ])

    with tabs[0]:
        render_fields_tab(result)
    with tabs[1]:
        render_tables_tab(result)
    with tabs[2]:
        render_elements_tab(result)
    with tabs[3]:
        render_degraded_blocks_tab(result)
    with tabs[4]:
        render_corrections_tab(result)
    with tabs[5]:
        render_mismatches_tab(result)
    with tabs[6]:
        render_page_quality_tab(result)
    with tabs[7]:
        st.subheader("Export Structured JSON")
        export_data = result.model_dump()
        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"{result.filename.replace('.pdf', '')}_extraction.json",
            mime="application/json",
        )
        with st.expander("Preview JSON"):
            st.code(
                json_str[:3000] + ("\n... (truncated)" if len(json_str) > 3000 else ""),
                language="json",
            )


def main():
    st.title("📄 Mortgage Document AI Pipeline")
    st.markdown(
        "Upload one or more mortgage PDFs. The pipeline extracts structured data, detects corrections, "
        "flags mismatches, identifies stamps and signatures, and classifies document types — all concurrently."
    )

    settings = render_sidebar()
    api_key = settings["api_key"]
    max_workers = settings["max_workers"]

    if not api_key:
        st.error(
            "❌ No Groq API key configured. Add it in the sidebar or set the GROQ_API_KEY env var. "
            "Get a free key at https://console.groq.com"
        )
        st.stop()

    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Upload PDF documents (multiple supported)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more mortgage PDFs — Loan Estimates, Appraisals, Closing Disclosures, etc.",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more PDF files to begin.")
        with st.expander("ℹ️ How the pipeline works"):
            st.markdown("""
**Stage 1 — PDF Loading:** `PyMuPDF` renders each page as a high-res image. `pdfplumber` extracts native tables.

**Stage 2 — Vision Analysis (Groq + Llama 4 Scout):** Each page image is sent concurrently to Groq with the extracted text layer. The model handles OCR for scanned pages, detects stamps/signatures/watermarks, and extracts all fields with evidence.

**Stage 3 — Correction Resolution:** Pages labeled as ADDENDUM/CORRECTION supersede original figures. Rule: latest date wins; if undated, higher page number wins. Not a "last page wins" assumption.

**Stage 4 — Mismatch Detection:** Arithmetic totals are verified against their components. Mismatches are flagged and preserved as-stated (not silently recomputed).

**Stage 5 — Confidence Scoring:** Each field gets HIGH/MEDIUM/LOW confidence based on page quality, format validation, and extraction certainty.
            """)
        return

    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if len(uploaded_files) > 5:
            st.warning("⚠️ Maximum 5 PDFs at once to respect API rate limits.")
            uploaded_files = uploaded_files[:5]

        tmp_paths = []
        filenames = []
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.read())
                tmp_paths.append(tmp.name)
            filenames.append(uf.name)

        orchestrator = DocumentOrchestrator(
            api_key=api_key,
            max_workers=max_workers,
            model=settings.get("model", "meta-llama/llama-4-scout-17b-16e-instruct"),
        )

        st.markdown("---")
        st.subheader("Processing Status")
        progress_bars = {}
        progress_texts = {}
        for i, fname in enumerate(filenames):
            st.markdown(f"**{fname}**")
            progress_bars[i] = st.progress(0.0)
            progress_texts[i] = st.empty()

        # Documents are processed sequentially on the main thread so that
        # Streamlit progress bars can be updated (NoSessionContext fix).
        # Page-level concurrency still runs inside process_single_pdf.
        results = []
        for i, (pdf_path, fname) in enumerate(zip(tmp_paths, filenames)):
            def make_callback(idx):
                def callback(pct, msg):
                    progress_bars[idx].progress(min(float(pct), 1.0))
                    progress_texts[idx].caption(msg)
                return callback

            try:
                result = orchestrator.process_single_pdf(
                    pdf_path, fname, make_callback(i)
                )
            except Exception as exc:
                full_tb = traceback.format_exc()
                result = ExtractionResult(
                    document_id=f"err-{i}",
                    filename=fname,
                    document_type="Unknown",
                    document_type_confidence="low",
                    document_type_reason=f"{type(exc).__name__}: {str(exc)}\n\n{full_tb}",
                    fields={},
                    tables=[],
                    detected_elements=[],
                    degraded_text_blocks=[],
                    corrections_applied=[],
                    mismatches=[],
                    page_analyses=[],
                    processing_time_seconds=0,
                    pages_processed=0,
                )
            results.append(result)

        for tmp_path in tmp_paths:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        st.markdown("---")
        st.success(f"✅ Processed {len(results)} document(s) successfully.")

        if len(results) == 1:
            render_result(results[0], 0)
        else:
            doc_tabs = st.tabs([r.filename for r in results])
            for i, (tab, result) in enumerate(zip(doc_tabs, results)):
                with tab:
                    render_result(result, i)


if __name__ == "__main__":
    main()