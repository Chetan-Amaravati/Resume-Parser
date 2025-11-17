import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

import streamlit as st
import pandas as pd
from pathlib import Path
from parser import parse_folder
from scoring import score_dataframe, summarize
from db_handler import ResumeDB
import io, json, ast
from typing import Any, Dict
from auth import show_login_register_page  # ✅ Login/Register UI

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Resume Parser", layout="wide")

# ------------------- THEME -------------------
st.markdown("""
    <style>
    .stApp { background-color: #fdf6e3; color: #3e2723; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #ede0d4; }
    div.stButton > button { background: #8d6e63; color: white; border-radius: 8px; padding: 0.6rem 1.2rem;
                            font-weight: bold; border: none; transition: 0.3s; }
    div.stButton > button:hover { background: #a1887f; transform: scale(1.03); }
    h1, h2, h3 { color: #4e342e !important; font-weight: 700; }
    .dataframe { background: #fffaf2; border-radius: 10px; padding: 10px;
                 box-shadow: 0px 2px 6px rgba(0,0,0,0.08); }
    hr { border: 1px solid #d7ccc8; }
    </style>
""", unsafe_allow_html=True)

# ------------------- MONGODB INITIALIZATION -------------------
@st.cache_resource
def init_db():
    return ResumeDB(mongo_uri="mongodb://localhost:27017", db_name="resume_db")

db = init_db()

# ------------------- LOGIN HANDLING -------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    show_login_register_page(db)
    st.stop()

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center;'>📄 Resume Parser with MongoDB</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#6d4c41;'> Upload resumes or use the included dataset. All data saved to MongoDB.</p>", unsafe_allow_html=True)

# ------------------- DB STATUS -------------------
if db.is_connected():
    st.sidebar.success("✓ MongoDB Connected")
else:
    st.sidebar.error("✗ MongoDB Not Connected")
    st.sidebar.warning("Run MongoDB locally with: mongod")

# ✅ Logout Button
if st.sidebar.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.session_state["user"] = {}
    st.rerun()

# ------------------- PATHS -------------------
skills_path  = Path("skills.json")
dataset_path = Path("dataset")
uploads_dir  = Path("uploads")

# ------------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.header("⚙️ Controls")
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
    save_to_db = st.checkbox("Auto-save to MongoDB", value=True, disabled=not db.is_connected())

    if st.button("🗑️ Clear Session Data", use_container_width=True):
        st.session_state["df"] = pd.DataFrame()
        st.rerun()

# ------------------- SESSION STATE -------------------
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "show_full_summary" not in st.session_state:
    st.session_state["show_full_summary"] = False
if "scored_df" not in st.session_state:
    st.session_state["scored_df"] = pd.DataFrame()

# ------------------- UTILITIES -------------------
def save_uploads(files) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        (uploads_dir / f.name).write_bytes(f.read())
    return uploads_dir

def parse_if_exists(folder: Path) -> pd.DataFrame:
    if folder and folder.exists() and any(folder.iterdir()):
        return parse_folder(folder, skills_path)
    return pd.DataFrame()

def _maybe_parse_json_like(x: Any):
    if isinstance(x, str) and x.strip().startswith(("{", "[")):
        s = x.strip()
        try: return json.loads(s)
        except Exception:
            try: return ast.literal_eval(s)
            except Exception: return x
    return x

def _format_contacts(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict): return str(d)
    keys_order = ["email", "phone", "linkedin", "github"]
    parts = []
    for k in keys_order:
        v = d.get(k)
        if v:
            label = "Phone" if k in ("phone",) else k.capitalize()
            parts.append(f"{label}: {v}")
    return " | ".join(parts) if parts else "-"

def _format_list(value: Any) -> str:
    value = _maybe_parse_json_like(value)
    if isinstance(value, list):
        items = [str(v) for v in value if v]
        return ", ".join(items[:10]) if items else "-"
    return str(value)

def _format_cgpa(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("CGPA", "").replace("GPA", "").strip()

def prettify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    cols = ["name", "contacts", "projects", "skills", "cgpa"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]
    if "contacts" in out: out["contacts"] = out["contacts"].apply(lambda v: _format_contacts(_maybe_parse_json_like(v)))
    if "projects" in out: out["projects"] = out["projects"].apply(_format_list)
    if "skills" in out: out["skills"] = out["skills"].apply(_format_list)
    if "cgpa" in out: out["cgpa"] = out["cgpa"].apply(_format_cgpa)
    out.insert(0, "No.", range(1, len(out)+1))
    return out

def smart_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    combined = pd.concat([existing, new], ignore_index=True)
    if "file" in combined.columns:
        combined = combined.drop_duplicates(subset=["file"], keep="first")
    return combined.reset_index(drop=True)

# ------------------- MAIN LAYOUT -------------------
tab1, tab2 = st.tabs(["Parsing Results", "Scoring & Analysis"])

# ------------------- TAB 1 -------------------
with tab1:
    st.subheader("🧾 Parsing Results")

    if parse_btn:
        frames = []
        if data_source == "Upload New Files" and uploaded:
            save_uploads(uploaded)
            frames.append(parse_if_exists(uploads_dir))
        elif data_source == "Use Included Dataset":
            if dataset_path.exists() and any(dataset_path.iterdir()):
                frames.append(parse_if_exists(dataset_path))
            else:
                st.warning("Dataset folder not found or empty.")
        elif data_source == "Load from MongoDB":
            if db.is_connected():
                db_df = db.get_resumes_dataframe()
                if not db_df.empty:
                    frames.append(db_df)
                else:
                    st.warning("No resumes found in MongoDB.")
        if frames:
            new_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            st.session_state["df"] = smart_concat(st.session_state["df"], new_df)
            if save_to_db and db.is_connected() and data_source != "Load from MongoDB":
                db.save_resumes_batch(st.session_state["df"])
            st.success(f"Parsed {len(st.session_state['df'])} resumes.")
        else:
            st.warning("No resumes parsed yet.")

    if not st.session_state["df"].empty:
        display_df = prettify_dataframe(st.session_state["df"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("### 📋 Choose details to include in Excel")
        available_columns = ["name", "contacts", "projects", "skills", "cgpa"]
        selected_columns = st.multiselect(
            "Select the details you want in the Excel file:",
            options=available_columns,
            default=available_columns
        )

        export_df = display_df[selected_columns]
        excel_data = io.BytesIO()
        with pd.ExcelWriter(excel_data, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, sheet_name="Resume Details", index=False)
        excel_data.seek(0)

        st.download_button(
            label="💾 Download Selected Details",
            data=excel_data,
            file_name="resume_details_selected.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ------------------- TAB 2: SCORING -------------------
with tab2:
    st.subheader("📊 Scoring & Skill Gap Analysis")

    # ------------------- SCORING PROCESS -------------------
    if score_btn and jd.strip() != "" and not st.session_state["df"].empty:
        with st.spinner("Analyzing skill gap..."):
            scored = score_dataframe(st.session_state["df"], jd)
            st.session_state["scored_df"] = scored.copy()

            if save_to_db and db.is_connected():
                for _, row in scored.iterrows():
                    db.save_scoring_result(
                        row["file"],
                        jd,
                        {"score": row["score"], "matched": row.get("matched", []),
                         "missing": row.get("missing", []), "project_domains": row.get("project_domains", {})}
                    )

    elif score_btn:
        st.warning("⚠️ Please provide a Job Description and parse resumes first.")

    # ------------------- FILTER SECTION (TOP) -------------------
    if not st.session_state["scored_df"].empty:

        st.markdown("### 🔎 Filter Results")

        scored_df = st.session_state["scored_df"].copy()

        # SCORE FILTER
        score_list = sorted(scored_df["score"].unique())
        selected_score = st.selectbox(
            "🎯 Filter by Score:",
            options=["All"] + score_list,
            index=0
        )

        filtered_df = scored_df.copy()
        if selected_score != "All":
            filtered_df = filtered_df[filtered_df["score"] == selected_score]

        # SKILL FILTER
        st.subheader("🛠️ Filter by Required Skills")

        all_required_skills = set()
        for x in scored_df["matched"]:
            if isinstance(x, list):
                all_required_skills.update(x)

        selected_skills = st.multiselect(
            "Select skills to filter candidates (must contain ALL selected skills):",
            sorted(list(all_required_skills)),
        )

        if selected_skills:
            filtered_df = filtered_df[
                filtered_df["matched"].apply(
                    lambda lst: all(skill in lst for skill in selected_skills)
                )
            ]

        # ------------------- SHOW FILTERED RESULTS -------------------
        st.markdown("### 📊 Filtered Scoring Results")

        display_df = filtered_df.copy()
        for c in ("matched", "missing"):
            if c in display_df.columns:
                display_df[c] = display_df[c].apply(_format_list)

        display_df.insert(0, "No.", range(1, len(display_df) + 1))
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        # Restored original domain display (old feature)
        if not filtered_df.empty:
            first_row = filtered_df.iloc[0]
        if "project_domains" in first_row:
            st.write("🔍 Project Domains:", first_row.get("project_domains", "-"))


# ------------------- RESUME SUMMARY -------------------
st.subheader("🧠 Resume Summary")
if not st.session_state["df"].empty:
    resume_files = st.session_state["df"]["file"].tolist()
    selected_file = st.selectbox("Select a Resume to Summarize", resume_files, index=0)
    selected_row = st.session_state["df"][st.session_state["df"]["file"] == selected_file].iloc[0]

    st.write(f"**Selected Resume: {selected_row.get('file', '-')}**")
    brief_details = []
    if "name" in selected_row:
        brief_details.append(f"**Name**: {selected_row['name']}")
    if "contacts" in selected_row:
        brief_details.append(f"**Contacts**: {_format_contacts(selected_row['contacts'])}")
    if "skills" in selected_row:
        brief_details.append(f"**Skills**: {_format_list(selected_row['skills'])}")

    raw_text = str(selected_row.get("raw_text", ""))
    if raw_text.strip():
        brief_summary = summarize(raw_text, max_sentences=1)
        brief_details.append(f"**Summary**: {brief_summary.split('.')[0]}.")

    for d in brief_details:
        st.write(d)

    if st.button("See More" if not st.session_state["show_full_summary"] else "See Less"):
        st.session_state["show_full_summary"] = not st.session_state["show_full_summary"]
        st.rerun()

    if st.session_state["show_full_summary"]:
        full_details = brief_details.copy()
        if "projects" in selected_row:
            full_details.append(f"**Projects**: {_format_list(selected_row['projects'])}")
        if "cgpa" in selected_row:
            full_details.append(f"**CGPA**: {_format_cgpa(selected_row['cgpa'])}")
        if raw_text.strip():
            full_summary = summarize(raw_text, max_sentences=3)
            full_details.append(f"**Summary**: {full_summary}")
        for d in full_details:
            st.write(d)
else:
    st.info("Parse resumes to view summaries.")

# ------------------- FOOTER -------------------
st.divider()
st.caption("✨ Features: MongoDB Storage • Customizable Excel Export • Filterable Scores • Skill Gap Analysis • Secure Login System")
