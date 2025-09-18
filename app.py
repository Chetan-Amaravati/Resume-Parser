# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from parser import parse_folder
from scoring import score_dataframe, summarize

import json, ast, shutil
from typing import Any, Dict

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Offline Resume Parser", layout="wide")

# ------------------- THEME (Beige + Brown) -------------------
st.markdown("""
    <style>
    .stApp { background-color: #fdf6e3; color: #3e2723; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #ede0d4; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p { color: #3e2723; }
    div.stButton > button { background: #8d6e63; color: white; border-radius: 8px; padding: 0.6rem 1.2rem;
                            font-weight: bold; border: none; transition: 0.3s; }
    div.stButton > button:hover { background: #a1887f; transform: scale(1.03); }
    h1, h2, h3 { color: #4e342e !important; font-weight: 700; }
    .dataframe { background: #fffaf2; border-radius: 10px; padding: 10px;
                 box-shadow: 0px 2px 6px rgba(0,0,0,0.08); }
    .stAlert > div { border-radius: 10px; padding: 0.8rem; }
    hr { border: 1px solid #d7ccc8; }
    a { color: #6d4c41 !important; text-decoration: none; font-weight: 500; }
    a:hover { text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center;'>📄 Resume Parser</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#6d4c41;'>Works fully offline. Upload resumes or use the included dataset.</p>", unsafe_allow_html=True)

# ------------------- PATHS -------------------
skills_path  = Path("skills.json")
dataset_path = Path("dataset")
uploads_dir  = Path("uploads")   # where we save drag&drop files

# ------------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.header("⚙️ Controls")
    use_dataset = st.checkbox("Use included dataset", value=True)
    jd = st.text_area("Job Description", height=140,
                      placeholder="Looking for a Python developer with Flask, Docker, AWS...")
    uploaded = st.file_uploader("Upload Resume(s)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    st.divider()
    parse_btn = st.button("🔍 Parse Resume(s)", use_container_width=True)
    score_btn = st.button("📈 Score & Skill Gap", use_container_width=True)

# ------------------- SESSION STATE -------------------
# cumulative parsed DataFrame (we'll append new parses)
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

# ------------------- UTILITIES -------------------
def save_uploads(files) -> Path:
    """
    Save uploaded files to uploads/ (appends; does NOT clear).
    Returns the folder path.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        (uploads_dir / f.name).write_bytes(f.read())
    return uploads_dir

def parse_if_exists(folder: Path) -> pd.DataFrame:
    """Parse a folder only if it exists and has files; else return empty df."""
    if folder and folder.exists() and any(folder.iterdir()):
        return parse_folder(folder, skills_path)
    return pd.DataFrame()

# ---------- Pretty printing helpers to avoid [object Object] ----------
JSON_LIKE_STARTS = ("{", "[")

def _maybe_parse_json_like(x: Any):
    if isinstance(x, str) and x.strip().startswith(JSON_LIKE_STARTS):
        s = x.strip()
        try: return json.loads(s)
        except Exception:
            try: return ast.literal_eval(s)
            except Exception: return x
    return x

def _format_contacts(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict): return str(d)
    keys_order = ["email", "phone", "mobile", "linkedin", "github", "portfolio", "website"]
    parts = []
    for k in keys_order:
        v = d.get(k)
        if v:
            label = "Phone" if k in ("phone", "mobile") else k.capitalize()
            parts.append(f"{label}: {v}")
    if not parts:
        parts = [f"{k}: {v}" for k, v in list(d.items())[:6]]
    return " | ".join(parts)

def _format_experience_item(item: Any) -> str:
    if isinstance(item, dict):
        company  = item.get("company") or item.get("organization") or item.get("org") or item.get("employer") or ""
        role     = item.get("role") or item.get("position") or item.get("title") or item.get("designation") or ""
        duration = item.get("duration") or item.get("years") or ""
        start    = item.get("start") or item.get("from") or ""
        end      = item.get("end") or item.get("to") or ""
        when     = duration or (f"{start}–{end}" if (start or end) else "")
        piece    = " ".join(p for p in [role, "@", company] if p).strip()
        if when: piece = f"{piece} ({when})" if piece else when
        return piece or str(item)
    return str(item)

def _format_experience(value: Any, max_items: int = 3) -> str:
    value = _maybe_parse_json_like(value)
    if isinstance(value, list):
        items = [_format_experience_item(v) for v in value if v is not None]
        items = [s for s in items if s]
        if not items: return "-"
        shown = items[:max_items]
        more  = f"  (+{len(items)-max_items} more)" if len(items) > max_items else ""
        return " • ".join(shown) + more
    if isinstance(value, dict):
        return _format_experience_item(value)
    return str(value)

def _format_confidence(d: Any, max_items: int = 4) -> str:
    d = _maybe_parse_json_like(d)
    if isinstance(d, dict):
        try: items = sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)
        except Exception: items = list(d.items())
        items = [f"{k}:{round(float(v),2) if str(v).replace('.','',1).isdigit() else v}" for k, v in items[:max_items]]
        more  = f"  (+{len(d)-max_items} more)" if len(d) > max_items else ""
        return ", ".join(items) + more
    return str(d)

def _format_list(value: Any, max_items: int = 10) -> str:
    value = _maybe_parse_json_like(value)
    if isinstance(value, list):
        items = [str(v) for v in value if v is not None]
        shown = items[:max_items]
        more  = f"  (+{len(items)-max_items} more)" if len(items) > max_items else ""
        return ", ".join(shown) + more if shown else "-"
    return str(value)

def prettify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with nested objects rendered into readable text + add 1-based #."""
    if df.empty: return df
    out = df.copy()

    if "contacts" in out.columns:
        out["contacts"] = out["contacts"].apply(lambda v: _format_contacts(_maybe_parse_json_like(v)))
    if "experience" in out.columns:
        out["experience"] = out["experience"].apply(_format_experience)
    if "confidence" in out.columns:
        out["confidence"] = out["confidence"].apply(_format_confidence)
    # prettify any other list/dict columns (not raw_text)
    for col in out.columns:
        if col in ("raw_text", "contacts", "experience", "confidence"): continue
        if out[col].apply(lambda x: isinstance(_maybe_parse_json_like(x), (list, dict))).any():
            out[col] = out[col].apply(_format_list)

    # drop heavy text from table, but it stays in session df for summary/scoring
    out = out.drop(columns=[c for c in out.columns if c == "raw_text"], errors="ignore")

    # add 1-based numbering
    out.insert(0, "No.", range(1, len(out) + 1))
    return out

def smart_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Append new rows to existing and drop duplicates intelligently.
    Primary key: 'file'. If 'contacts' contains an email, that helps too.
    """
    if existing is None or existing.empty:
        combined = new.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)

    # prefer dedupe by 'file'
    if "file" in combined.columns:
        combined = combined.drop_duplicates(subset=["file"], keep="first")

    # optional: dedupe by primary email if present (contacts parsed text may vary)
    if "contacts" in combined.columns:
        # Try to extract an email-like token from the prettified contacts
        def primary_email(txt: Any) -> str:
            s = str(txt)
            # crude pick: find the first email-ish span
            import re
            m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s)
            return m.group(0).lower() if m else ""
        combined["_email_key"] = combined["contacts"].apply(primary_email)
        combined = combined.drop_duplicates(subset=["_email_key", "file"], keep="first").drop(columns="_email_key")

    combined = combined.reset_index(drop=True)
    return combined

# ------------------- MAIN LAYOUT -------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧾 Parsing Results")

    if parse_btn:
        # 1) Save any newly uploaded files (append, do NOT clear)
        if uploaded:
            save_uploads(uploaded)
            st.toast(f"Saved {len(uploaded)} uploaded file(s) to /uploads", icon="📥")

        # 2) Decide which folders to parse this click:
        #    - If dataset is ticked -> include dataset
        #    - Always include uploads if uploads_dir has files
        frames = []
        if use_dataset and dataset_path.exists() and any(dataset_path.iterdir()):
            st.info(f"Including dataset: {dataset_path.resolve()}")
            frames.append(parse_if_exists(dataset_path))
        if uploads_dir.exists() and any(uploads_dir.iterdir()):
            st.info(f"Including uploads: {uploads_dir.resolve()}")
            frames.append(parse_if_exists(uploads_dir))

        if len(frames) == 0:
            st.warning("No resumes found. Upload files or enable 'Use included dataset'.")
        else:
            new_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

            # cumulative append into session + dedupe
            st.session_state["df"] = smart_concat(st.session_state["df"], new_df)

            st.success(f"Parsed {len(st.session_state['df'])} total resumes (cumulative).")

            display_df = prettify_dataframe(st.session_state["df"])

            column_config = {}
            if "experience" in display_df.columns:
                column_config["experience"] = st.column_config.TextColumn("experience", width="large")
            if "file" in display_df.columns:
                column_config["file"] = st.column_config.TextColumn("file", width="medium")

            st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)

with col2:
    st.subheader("🧠 Resume Summary")
    if not st.session_state["df"].empty:
        row = st.session_state["df"].iloc[0]
        st.write(f"**Sample Resume: {row.get('file','-')}**")
        summary = summarize(row.get("raw_text", ""), max_sentences=3)
        st.write(summary)

# ------------------- SCORING SECTION -------------------
st.divider()
st.subheader("📊 Scoring & Skill Gap Analysis")

if score_btn:
    if jd.strip() == "" or st.session_state["df"].empty:
        st.warning("⚠️ Please provide a Job Description and parse resumes first.")
    else:
        scored = score_dataframe(st.session_state["df"], jd)

        # prettify matched/missing columns if they are lists
        scored_display = scored.copy()
        for c in ("matched", "missing"):
            if c in scored_display.columns:
                scored_display[c] = scored_display[c].apply(_format_list)

        # 1-based numbering
        scored_display.insert(0, "No.", range(1, len(scored_display) + 1))

        st.dataframe(scored_display, use_container_width=True, hide_index=True)

        if not scored_display.empty:
            srow = scored_display.iloc[0]
            st.success(f"🏆 Top Match: {srow['file']} — Score: {srow['score']}")
            st.write("✅ Matched Skills:", srow.get("matched", "-"))
            st.write("❌ Missing Skills:", srow.get("missing", "-"))

# ------------------- FOOTER -------------------
st.divider()
st.caption("✨ Features: Section Confidence • Offline Summaries • Multi-language (English/Kannada) • Word Cloud (see README)")
