# app.py
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
from auth import show_login_register_page

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
    .project-cell { white-space: pre-line; font-size: 0.9em; }
    </style>
""", unsafe_allow_html=True)

# ------------------- MONGODB -------------------
@st.cache_resource
def init_db():
    return ResumeDB(mongo_uri="mongodb://localhost:27017", db_name="resume_db")

db = init_db()

# ------------------- LOGIN -------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    show_login_register_page(db)
    st.stop()

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center;'>Resume Parser and Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#6d4c41;'>Upload resumes or use the included dataset. All data saved to MongoDB.</p>", unsafe_allow_html=True)

# DB Status
if db.is_connected():
    st.sidebar.success("MongoDB Connected")
else:
    st.sidebar.error("MongoDB Not Connected")
    st.sidebar.warning("Run: `mongod`")

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.session_state["user"] = {}
    st.rerun()

# ------------------- PATHS -------------------
skills_path  = Path("skills.json")
dataset_path = Path("dataset")
uploads_dir  = Path("uploads")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("Controls")
    data_source = st.radio("Data Source", ["Upload New Files", "Use Included Dataset", "Load from MongoDB"], index=0)
    jd = st.text_area("Job Role", height=140, placeholder="Looking for Python developer, Data scientist...")

    uploaded = None
    if data_source == "Upload New Files":
        uploaded = st.file_uploader("Upload Resume(s)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    st.divider()
    parse_btn = st.button("Parse Resume(s)", use_container_width=True)
    score_btn = st.button("Score & Skill Gap", use_container_width=True)

    st.divider()
    save_to_db = st.checkbox("Auto-save to MongoDB", value=True, disabled=not db.is_connected())

    if st.button("Clear Session Data", use_container_width=True):
        st.session_state["df"] = pd.DataFrame()
        st.session_state["scored_df"] = pd.DataFrame()
        st.rerun()

# ------------------- SESSION STATE -------------------
if "df" not in st.session_state: st.session_state["df"] = pd.DataFrame()
if "show_full_summary" not in st.session_state: st.session_state["show_full_summary"] = False
if "scored_df" not in st.session_state: st.session_state["scored_df"] = pd.DataFrame()

# ------------------- UTILS -------------------
def save_uploads(files) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for f in uploads_dir.glob("*"): f.unlink()
    for f in files: (uploads_dir / f.name).write_bytes(f.read())
    return uploads_dir

def parse_if_exists(folder: Path) -> pd.DataFrame:
    return parse_folder(folder, skills_path) if folder.exists() and any(folder.iterdir()) else pd.DataFrame()

def _maybe_parse_json_like(x: Any):
    if isinstance(x, str) and x.strip().startswith(("{", "[")):
        try: return json.loads(x.strip())
        except: 
            try: return ast.literal_eval(x.strip())
            except: return x
    return x

def _format_contacts(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict): return str(d)
    keys = ["email", "phone", "linkedin", "github"]
    parts = []
    for k in keys:
        v = d.get(k)
        if v: parts.append(f"{k.capitalize() if k != 'phone' else 'Phone'}: {v}")
    return " | ".join(parts) or "-"

def _format_list(value: Any) -> str:
    value = _maybe_parse_json_like(value)
    if isinstance(value, list):
        return ", ".join([str(v) for v in value if v][:10]) or "-"
    return str(value)

def _format_cgpa(value: Any) -> str:
    return "-" if not value else str(value).replace("CGPA", "").replace("GPA", "").strip()

def _format_project_titles(project_domains: Dict) -> str:
    if not project_domains or not isinstance(project_domains, dict):
        return "No projects"
    titles = []
    for title in project_domains.keys():
        titles.append(f"• {title}")
    return "\n".join(titles)

def prettify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    cols = [c for c in ["name", "contacts", "projects", "skills", "cgpa"] if c in out.columns]
    out = out[cols]
    if "contacts" in out: out["contacts"] = out["contacts"].apply(lambda v: _format_contacts(_maybe_parse_json_like(v)))
    if "projects" in out: out["projects"] = out["projects"].apply(_format_list)
    if "skills" in out: out["skills"] = out["skills"].apply(_format_list)
    if "cgpa" in out: out["cgpa"] = out["cgpa"].apply(_format_cgpa)
    out.insert(0, "No.", range(1, len(out)+1))
    return out

def smart_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty: return new.copy()
    combined = pd.concat([existing, new], ignore_index=True)
    if "file" in combined.columns: combined = combined.drop_duplicates(subset=["file"], keep="first")
    return combined.reset_index(drop=True)

# ------------------- TABS -------------------
tab1, tab2 = st.tabs(["Parsing Results", "Scoring & Analysis"])

# ------------------- TAB 1: PARSING -------------------
with tab1:
    st.subheader("Parsing Results")
    if parse_btn:
        frames = []
        if data_source == "Upload New Files" and uploaded:
            save_uploads(uploaded)
            frames.append(parse_if_exists(uploads_dir))
        elif data_source == "Use Included Dataset":
            if dataset_path.exists(): frames.append(parse_if_exists(dataset_path))
            if db.is_connected(): frames.append(db.get_resumes_dataframe())
        elif data_source == "Load from MongoDB" and db.is_connected():
            frames.append(db.get_resumes_dataframe())

        if frames:
            new_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            st.session_state["df"] = new_df if data_source in ["Upload New Files", "Load from MongoDB"] else smart_concat(st.session_state["df"], new_df)
            if save_to_db and db.is_connected() and data_source != "Load from MongoDB":
                db.save_resumes_batch(st.session_state["df"])
            st.success(f"Parsed {len(st.session_state['df'])} resumes.")
        else:
            st.warning("No resumes parsed.")

    if not st.session_state["df"].empty:
        display_df = prettify_dataframe(st.session_state["df"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        cols = ["name", "contacts", "projects", "skills", "cgpa"]
        selected = st.multiselect("Export Columns", options=cols, default=cols)
        export_df = display_df[selected]
        excel = io.BytesIO()
        export_df.to_excel(excel, index=False); excel.seek(0)
        st.download_button("Download Excel", excel, "resume_details.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------- TAB 2: SCORING + FILTER + PROJECTS IN TABLE -------------------
with tab2:
    st.subheader("Scoring & Skill Gap Analysis")

    # ---------- SCORE ----------
    if score_btn and jd.strip() and not st.session_state["df"].empty:
        with st.spinner("Scoring..."):
            scored = score_dataframe(st.session_state["df"], jd)
            st.session_state["scored_df"] = scored.copy()
            if save_to_db and db.is_connected():
                for _, r in scored.iterrows():
                    db.save_scoring_result(r["file"], jd, {
                        "score": r["score"], "matched": r.get("matched", []),
                        "missing": r.get("missing", []), "project_domains": r.get("project_domains", {})
                    })
    elif score_btn:
        st.warning("Add JD and parse resumes first.")

    # ---------- FILTER SECTION (2 OPTIONS) ----------
    if "scored_df" in st.session_state and not st.session_state["scored_df"].empty:
        raw = st.session_state["scored_df"]

        # Extract filter options
        all_skills = sorted({s for row in raw["matched"] for s in row})
        unique_scores = sorted(raw["score"].unique(), reverse=True)
        score_options = ["All"] + [str(s) for s in unique_scores]

        st.markdown("### Filter")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                selected_score = st.selectbox(
                    "Filter by Score",
                    options=score_options,
                    index=0,
                    key="filter_score"
                )
            with col2:
                required_skills = st.multiselect(
                    "Filter by Required Skills",
                    options=all_skills,
                    default=[],
                    key="filter_skills"
                )

            if st.button("Reset Filters", key="reset_filters"):
                st.session_state["filter_score"] = "All"
                st.session_state["filter_skills"] = []
                st.rerun()

        # === APPLY FILTERS ===
        filtered = raw.copy()
        if selected_score != "All":
            filtered = filtered[filtered["score"] == int(selected_score)]
        if required_skills:
            filtered = filtered[filtered["matched"].apply(
                lambda skills: all(skill in skills for skill in required_skills)
            )]

        # === DISPLAY TABLE WITH PROJECT TITLES INSIDE ===
        display_df = filtered[["file", "score", "matched", "missing", "project_domains"]].copy()
        display_df["matched"] = display_df["matched"].apply(_format_list)
        display_df["missing"] = display_df["missing"].apply(_format_list)
        display_df["project_titles"] = display_df["project_domains"].apply(_format_project_titles)
        display_df = display_df.drop(columns=["project_domains"])
        display_df.insert(0, "No.", range(1, len(display_df) + 1))

        # Rename for display
        display_df = display_df.rename(columns={"project_titles": "Projects"})

        st.markdown("### Filtered Candidates")
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Projects": st.column_config.TextColumn(
                    "Projects",
                    width="medium",
                    help="Click to expand project titles"
                )
            }
        )

        # === EXPORT ===
        if not display_df.empty:
            csv = display_df.to_csv(index=False).encode()
            st.download_button(
                "Export Filtered Results",
                data=csv,
                file_name=f"filtered_score_{selected_score}.csv",
                mime="text/csv"
            )

        # === TOP SCORER CARD + ONLY THEIR PROJECT DOMAINS (JSON) ===
        if not display_df.empty:
            top_raw = filtered.iloc[0]
            top_display = display_df.iloc[0]

            st.markdown("---")
            st.success(f"**Top Scorer:** {top_display['file']} – Score: **{top_display['score']}**")
            st.write(f"**Matched:** {top_display['matched']}")
            st.write(f"**Missing:** {top_display['missing']}")

            st.markdown("**Projects:**")
            st.markdown(top_display["Projects"].replace("• ", "- "), unsafe_allow_html=True)

            # Only show JSON domains for top scorer
            st.markdown("**Project Domains (JSON):**")
            top_domains = top_raw.get("project_domains", {})
            st.code(json.dumps(top_domains, indent=2, ensure_ascii=False), language="json")

    else:
        st.info("Run **Score & Skill Gap** to see results.")

# ------------------- SUMMARY -------------------
st.subheader("Resume Summary")
if not st.session_state["df"].empty:
    file = st.selectbox("Select Resume", st.session_state["df"]["file"])
    row = st.session_state["df"][st.session_state["df"]["file"] == file].iloc[0]
    st.write(f"**{row.get('file', '-')}**")
    st.write(f"**Name:** {row.get('name', '-')}")
    st.write(f"**Contacts:** {_format_contacts(row.get('contacts', {}))}")
    st.write(f"**Skills:** {_format_list(row.get('skills', []))}")
    if row.get("raw_text", ""):
        st.write(f"**Summary:** {summarize(str(row['raw_text']), max_sentences=1).split('.')[0]}.")
    if st.button("See More" if not st.session_state["show_full_summary"] else "See Less"):
        st.session_state["show_full_summary"] = not st.session_state["show_full_summary"]
        st.rerun()
    if st.session_state["show_full_summary"]:
        st.write(f"**Projects:** {_format_list(row.get('projects', []))}")
        st.write(f"**CGPA:** {_format_cgpa(row.get('cgpa'))}")
        st.write(f"**Full Summary:** {summarize(str(row['raw_text']), max_sentences=3)}")
else:
    st.info("Parse resumes to view summary.")

# ------------------- FOOTER -------------------
st.divider()
st.caption("Filter: Score + Skills • Project Titles in Table • Top Scorer Only • Clean UI")