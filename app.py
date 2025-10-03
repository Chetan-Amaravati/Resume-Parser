# app.py
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disables file watching to avoid Torch conflicts
try:
    import torch
    torch.classes.__path__ = []  # Prevents the '__path__._path' instantiation error
except ImportError:
    pass  # If Torch isn't imported yet, it's fine

import streamlit as st
import pandas as pd
from pathlib import Path
from parser import parse_folder
from scoring import score_dataframe, summarize
from db_handler import ResumeDB

import json, ast, shutil
from typing import Any, Dict

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Resume Parser", layout="wide")

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

# ------------------- MONGODB INITIALIZATION -------------------
@st.cache_resource
def init_db():
    """Initialize MongoDB connection (cached to avoid reconnecting)"""
    return ResumeDB(mongo_uri="mongodb://localhost:27017", db_name="resume_db")

db = init_db()

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center;'>📄 Resume Parser with MongoDB</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#6d4c41;'> Upload resumes or use the included dataset. All data saved to MongoDB.</p>", unsafe_allow_html=True)

# ------------------- DB STATUS -------------------
if db.is_connected():
    st.sidebar.success("✓ MongoDB Connected")
else:
    st.sidebar.error("✗ MongoDB Not Connected")
    st.sidebar.warning("Install MongoDB and run: `mongod`")

# ------------------- PATHS -------------------
skills_path  = Path("skills.json")
dataset_path = Path("dataset")
uploads_dir  = Path("uploads")

# ------------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Upload New Files", "Use Included Dataset", "Load from MongoDB"],
        index=0
    )
    
    jd = st.text_area("Job Description", height=140,
                      placeholder="Looking for a Python developer with Flask, Docker, AWS...")
    
    uploaded = None
    if data_source == "Upload New Files":
        uploaded = st.file_uploader("Upload Resume(s)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    st.divider()
    parse_btn = st.button("🔍 Parse Resume(s)", use_container_width=True)
    score_btn = st.button("📈 Score & Skill Gap", use_container_width=True)
    
    st.divider()
    st.subheader("💾 Database Options")
    save_to_db = st.checkbox("Auto-save to MongoDB", value=True, 
                             disabled=not db.is_connected())
    
    if st.button("🗑️ Clear Session Data", use_container_width=True):
        st.session_state["df"] = pd.DataFrame()
        st.rerun()

# ------------------- SESSION STATE -------------------
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

# ------------------- UTILITIES -------------------
def save_uploads(files) -> Path:
    """Save uploaded files to uploads/ directory"""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        (uploads_dir / f.name).write_bytes(f.read())
    return uploads_dir

def parse_if_exists(folder: Path) -> pd.DataFrame:
    """Parse a folder only if it exists and has files"""
    if folder and folder.exists() and any(folder.iterdir()):
        return parse_folder(folder, skills_path)
    return pd.DataFrame()

# ---------- Pretty printing helpers ----------
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
        company  = item.get("company") or item.get("organization") or ""
        role     = item.get("role") or item.get("position") or ""
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

def _format_domains(d: Dict) -> str:
    """Format project domains with confirmation info"""
    if isinstance(d, dict):
        proj_info = next(iter(d.values())) if d else {"domains": [], "confirmed": False, "confidence": 0.0}
        domains = ", ".join(proj_info["domains"]) if proj_info["domains"] else "-"
        conf_str = f" (Conf: {proj_info['confidence']}, {'✓ Confirmed' if proj_info['confirmed'] else '❌ Unconfirmed'})"
        return f"{domains}{conf_str}"
    return str(d)

def prettify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with nested objects rendered into readable text"""
    if df.empty: return df
    out = df.copy()

    if "contacts" in out.columns:
        out["contacts"] = out["contacts"].apply(lambda v: _format_contacts(_maybe_parse_json_like(v)))
    if "experience" in out.columns:
        out["experience"] = out["experience"].apply(_format_experience)
    if "confidence" in out.columns:
        out["confidence"] = out["confidence"].apply(_format_confidence)
    if "project_domains" in out.columns:
        out["project_domains"] = out["project_domains"].apply(_format_domains)
    
    for col in out.columns:
        if col in ("raw_text", "contacts", "experience", "confidence", "project_domains"): continue
        if out[col].apply(lambda x: isinstance(_maybe_parse_json_like(x), (list, dict))).any():
            out[col] = out[col].apply(_format_list)

    # drop raw_text from display
    out = out.drop(columns=[c for c in out.columns if c == "raw_text"], errors="ignore")
    
    # also drop MongoDB-specific fields from display
    out = out.drop(columns=[c for c in out.columns if c in ("uploaded_at", "last_updated", "_id")], errors="ignore")

    # add 1-based numbering
    out.insert(0, "No.", range(1, len(out) + 1))
    return out

def smart_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to existing and drop duplicates"""
    if existing is None or existing.empty:
        combined = new.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)

    if "file" in combined.columns:
        combined = combined.drop_duplicates(subset=["file"], keep="first")

    combined = combined.reset_index(drop=True)
    return combined

# ------------------- MAIN LAYOUT -------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧾 Parsing Results")

    if parse_btn:
        frames = []
        
        if data_source == "Upload New Files" and uploaded:
            # Save uploaded files
            save_uploads(uploaded)
            st.toast(f"Saved {len(uploaded)} uploaded file(s) to /uploads", icon="📥")
            frames.append(parse_if_exists(uploads_dir))
        
        elif data_source == "Use Included Dataset":
            if dataset_path.exists() and any(dataset_path.iterdir()):
                st.info(f"Including dataset: {dataset_path.resolve()}")
                frames.append(parse_if_exists(dataset_path))
            else:
                st.warning("Dataset folder not found or empty")
        
        elif data_source == "Load from MongoDB":
            if db.is_connected():
                with st.spinner("Loading resumes from MongoDB..."):
                    db_df = db.get_resumes_dataframe()
                    if not db_df.empty:
                        frames.append(db_df)
                        st.success(f"Loaded {len(db_df)} resumes from MongoDB")
                    else:
                        st.warning("No resumes found in MongoDB")
            else:
                st.error("MongoDB not connected")

        if len(frames) == 0:
            st.warning("No resumes found. Upload files or select a data source.")
        else:
            new_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            st.session_state["df"] = smart_concat(st.session_state["df"], new_df)
            
            # Save to MongoDB if enabled
            if save_to_db and db.is_connected() and data_source != "Load from MongoDB":
                with st.spinner("Saving to MongoDB..."):
                    saved_ids = db.save_resumes_batch(st.session_state["df"])
                    st.success(f"💾 Saved {len(saved_ids)} resumes to MongoDB")
            
            st.success(f"Parsed {len(st.session_state['df'])} total resumes (cumulative).")

            display_df = prettify_dataframe(st.session_state["df"])

            column_config = {}
            if "experience" in display_df.columns:
                column_config["experience"] = st.column_config.TextColumn("experience", width="large")
            if "file" in display_df.columns:
                column_config["file"] = st.column_config.TextColumn("file", width="medium")
            if "project_domains" in display_df.columns:
                column_config["project_domains"] = st.column_config.TextColumn("project_domains", width="large")

            st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)

with col2:
    st.subheader("🧠 Resume Summary")
    if not st.session_state["df"].empty:
        # Get unique resume filenames from the DataFrame
        resume_files = st.session_state["df"]["file"].tolist()
        selected_file = st.selectbox("Select a Resume to Summarize", resume_files, index=0)
        
        # Find the row for the selected resume
        selected_row = st.session_state["df"][st.session_state["df"]["file"] == selected_file].iloc[0]
        st.write(f"**Selected Resume: {selected_row.get('file', '-')}")
        
        # Get raw_text and handle type conversion
        raw_text = selected_row.get("raw_text", "")
        if not isinstance(raw_text, str):
            print(f"Debug: raw_text is {type(raw_text)} with value {raw_text} for file {selected_row.get('file', 'unknown')}")
            raw_text = str(raw_text) if raw_text is not None else ""
        summary = summarize(raw_text, max_sentences=3)
        st.write(summary)
    else:
        st.info("Parse resumes to see summary and selection options")

# ------------------- SCORING SECTION -------------------
st.divider()
st.subheader("📊 Scoring & Skill Gap Analysis")

if score_btn:
    if jd.strip() == "" or st.session_state["df"].empty:
        st.warning("⚠️ Please provide a Job Description and parse resumes first.")
    else:
        scored = score_dataframe(st.session_state["df"], jd)

        # Save scoring results to MongoDB
        if save_to_db and db.is_connected():
            with st.spinner("Saving scoring results..."):
                for _, row in scored.iterrows():
                    db.save_scoring_result(
                        row["file"], 
                        jd, 
                        {"score": row["score"], "matched": row.get("matched", []), "missing": row.get("missing", []), "project_domains": row.get("project_domains", {})}
                    )
                st.success("💾 Scoring results saved to MongoDB")

        # prettify matched/missing columns
        scored_display = scored.copy()
        for c in ("matched", "missing"):
            if c in scored_display.columns:
                scored_display[c] = scored_display[c].apply(_format_list)

        scored_display.insert(0, "No.", range(1, len(scored_display) + 1))

        st.dataframe(scored_display, use_container_width=True, hide_index=True)

        if not scored_display.empty:
            srow = scored_display.iloc[0]
            st.success(f"🏆 Top Match: {srow['file']} – Score: {srow['score']}")
            st.write("✅ Matched Skills:", srow.get("matched", "-"))
            st.write("❌ Missing Skills:", srow.get("missing", "-"))
            if "project_domains" in srow:
                st.write("🔍 Project Domains:", srow.get("project_domains", "-"))

# ------------------- FOOTER -------------------
st.divider()
st.caption("✨ Features: MongoDB Storage • Section Confidence • Offline Summaries • Multi-language (English/Kannada) • Project Domain Inference")