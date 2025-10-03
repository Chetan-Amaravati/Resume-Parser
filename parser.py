import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# PDF/DOCX loaders
import fitz  # PyMuPDF
from docx import Document

SECTION_HEADERS_EN = [
    "education","experience","work experience","projects","skills","certifications","summary","objective","contact","achievements"
]

# Minimal Kannada header variants (multi-language support)
SECTION_HEADERS_KN = [
    "ಶಿಕ್ಷಣ",  # education
    "ಅನುಭವ",  # experience
    "ಪ್ರಾಜೆಕ್ಟ್",  # projects (transliteration)
    "ಕೌಶಲ್ಯ",   # skills
    "ಸಾರಾಂಶ",   # summary
    "ಸಂಪರ್ಕ"    # contact
]

def load_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        with fitz.open(file_path) as doc:
            text = []
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    elif ext == ".docx":
        doc = Document(str(file_path))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return file_path.read_text(encoding="utf-8", errors="ignore")

def clean_text(text: str) -> str:
    text = text.replace("\x00"," ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_kannada(text: str) -> bool:
    # basic unicode range check
    return bool(re.search(r"[\u0C80-\u0CFF]", text))

def split_sections(text: str) -> Dict[str, str]:
    # Build header patterns in both languages
    headers = SECTION_HEADERS_EN + SECTION_HEADERS_KN
    pattern = r"(?mi)^(?P<header>(" + "|".join([re.escape(h) for h in headers]) + r"))\s*:?\s*$"
    sections = {}
    last = 0
    current = "header"
    matches = list(re.finditer(pattern, text))
    if not matches:
        sections["body"] = text
        return sections
    for i, m in enumerate(matches):
        if i == 0:
            pre = text[:m.start()].strip()
            if pre:
                sections["intro"] = pre
        if i+1 < len(matches):
            chunk = text[m.end():matches[i+1].start()].strip()
        else:
            chunk = text[m.end():].strip()
        sections[m.group("header").lower()] = chunk
    return sections

def extract_name(text: str) -> Tuple[str, float]:
    # Heuristic: first line with 2 words and > 3 chars each, no all caps noise tokens like RESUME/CV
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:5]:
        if len(l.split()) in (2,3) and not re.search(r"(resume|cv|curriculum vitae)", l, re.I):
            if re.match(r"[A-Za-zÀ-ÖØ-öø-ÿ'.-]+\s+[A-Za-zÀ-ÖØ-öø-ÿ'.-]+", l):
                return l, 0.85
    # fallback: email username
    m = re.search(r"([a-z0-9._%+-]+)@([a-z0-9.-]+)", text, re.I)
    if m:
        cand = m.group(1).replace(".", " ").title()
        return cand, 0.5
    return "", 0.0

def extract_contacts(text: str) -> Tuple[Dict[str,str], float]:
    phone = re.search(r"(\+91[- ]?)?\b[6-9]\d{9}\b", text)
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    linkedin = re.search(r"(linkedin\.com\/in\/[A-Za-z0-9\-_/]+)", text, re.I)
    score = 0.0
    out = {}
    if phone:
        out["phone"] = phone.group(0)
        score += 0.35
    if email:
        out["email"] = email.group(0)
        score += 0.35
    if linkedin:
        out["linkedin"] = linkedin.group(1)
        score += 0.3
    return out, min(score, 0.99)

def load_skills_dict(skills_path: Path) -> List[str]:
    return json.loads(skills_path.read_text(encoding="utf-8"))

def extract_skills(text: str, skills_list: List[str]) -> Tuple[List[str], float]:
    found = set()
    low = text.lower()
    for sk in skills_list:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(sk.lower())}(?![A-Za-z0-9])", low):
            found.add(sk)
    score = min(0.3 + 0.7*len(found)/max(1,len(skills_list)//5), 0.98) if found else 0.1
    return sorted(found), score

def extract_education(text: str) -> Tuple[str, float]:
    # capture common degree patterns
    edu = re.findall(r"(B\.?Tech|B\.?E\.?|M\.?Tech|M\.?S\.?|BSc|MSc|Diploma|Bachelor|Master).{0,80}(IIT|NIT|University|College|Institute|IISc|BITS).{0,40}\d{4}", text, re.I)
    if edu:
        return "\n".join(["".join(e) for e in edu]), 0.8
    # fallback: Education section content if present
    m = re.search(r"(?is)education\s*:?\s*(.+?)(\n[A-Z][A-Za-z ]{2,}\n|$)", text)
    if m:
        return m.group(1).strip(), 0.6
    return "", 0.0

def extract_projects(text: str) -> Tuple[List[str], float]:
    projects = re.findall(r"(?m)^\s*[-*•]\s*(.+)$", text)
    if projects:
        return projects[:10], 0.7
    # Enhanced fallback: Capture entire 'projects' section
    m = re.search(r"(?is)projects\s*:?\s*(.+?)(?=\n[A-Z][A-Za-z ]{2,}\n|\Z)", text)
    if m:
        lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
        return lines[:10], 0.55
    return [], 0.0

def extract_experience(text: str) -> Tuple[List[Dict[str,str]], float]:
    # naive bullet parsing with company and dates
    exp_blocks = []
    for block in re.split(r"(?i)\bexperience\b|work experience", text):
        if len(block) < 20: 
            continue
        bullets = re.findall(r"(?m)^\s*[-*•]\s*(.+)$", block)
        # company & dates
        comp = re.search(r"([A-Z][A-Za-z& ]{2,}),?\s+(Bengaluru|Hyderabad|Pune|Chennai|Mumbai|Delhi|Mysuru|Mangaluru|Udupi|Hubballi)?", block)
        dates = re.search(r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[–-]\s*(Present|\w+\s+\d{4}))", block, re.I)
        exp_blocks.append({
            "company": comp.group(1) if comp else "",
            "location": comp.group(2) if comp and comp.lastindex and comp.lastindex >= 2 else "",
            "dates": dates.group(1) if dates else "",
            "bullets": bullets[:6]
        })
    if exp_blocks:
        return exp_blocks[:3], 0.7
    return [], 0.0

def section_confidence_map(conf_dict: Dict[str, float]) -> Dict[str, int]:
    # convert to percentage
    return {k: int(round(v*100)) for k,v in conf_dict.items()}

def parse_file(path: Path, skills_path: Path) -> Dict:
    raw = load_text(path)
    text = clean_text(raw)
    sections = split_sections(text)
    skills_list = load_skills_dict(skills_path)

    name, c_name = extract_name(text)
    contacts, c_contact = extract_contacts(text)
    skills, c_skills = extract_skills(text, skills_list)
    education, c_edu = extract_education(text)
    projects, c_proj = extract_projects(text)
    experience, c_exp = extract_experience(text)

    conf = section_confidence_map({
        "name": c_name,
        "contact": c_contact,
        "skills": c_skills,
        "education": c_edu,
        "projects": c_proj,
        "experience": c_exp
    })

    projects_text = " ".join(projects)  # Combined text for scoring/inference

    return {
        "file": path.name,
        "name": name,
        "contacts": contacts,
        "education": education,
        "skills": skills,
        "projects": projects,
        "projects_text": projects_text,
        "experience": experience,
        "confidence": conf,
        "language": "Kannada" if is_kannada(text) else "English",
        "raw_text": text
    }

def parse_folder(folder: Path, skills_path: Path) -> pd.DataFrame:
    records = []
    for p in folder.glob("*"):
        if p.suffix.lower() not in (".pdf",".docx",".txt"):
            continue
        try:
            rec = parse_file(p, skills_path)
            records.append(rec)
        except Exception as e:
            records.append({"file": p.name, "error": str(e)})
    return pd.DataFrame(records)  # Fixed line