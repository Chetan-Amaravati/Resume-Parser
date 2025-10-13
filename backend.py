import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from parser import parse_file
import scoring
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
import shutil
import requests
from bs4 import BeautifulSoup
import spacy
from typing import List, Dict, Tuple

# Load skills.json path relative to backend.py location
SKILLS_JSON_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "skills.json"
with open(SKILLS_JSON_PATH) as f:
    SKILLS_VOCAB = json.load(f)

MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client['resume_db']
resumes_col = db['resumes']
domain_keywords_col = db['domain_keywords']

app = FastAPI(title="Dynamic Resume Parsing & Scoring Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

nlp = spacy.load("en_core_web_sm")

# Copied from scoring.py for consistency
COMMON_DOMAINS = [
    "web development", "mobile development", "machine learning", "data science", "artificial intelligence",
    "blockchain", "cybersecurity", "embedded systems", "cloud computing", "devops", "game development",
    "iot", "big data", "software engineering", "ui/ux design", "database management", "networking",
    "computer vision", "deep learning",
]

def web_search_extract_keywords(query: str, top_n=3) -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    snippets = []
    for g in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')[:top_n]:
        snippets.append(g.get_text())
    text_blob = " ".join(snippets)
    doc = nlp(text_blob)
    keywords = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2:
            keywords.add(chunk.text.lower())
    for ent in doc.ents:
        keywords.add(ent.text.lower())
    return list(keywords)

def confirm_domain_match(project_keywords: set, domain: str) -> Tuple[bool, float]:
    domain_toks = set(re.findall(r'\b[a-zA-Z]{3,}\b', domain.lower()))
    overlap = len(project_keywords & domain_toks)
    confidence = overlap / max(1, len(domain_toks))
    confirmed = confidence > 0.5
    return confirmed, confidence

def infer_project_domains(projects: List[str], use_web_search: bool = True) -> Dict[str, Dict[str, any]]:
    domains_per_project = {}
    
    for project in projects:
        if not project.strip():
            continue
        
        project_key = project[:50] + "..."
        
        # Local keywords
        local_keywords = set()
        doc = nlp(project)
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2:
                local_keywords.add(chunk.text.lower())
        for ent in doc.ents:
            local_keywords.add(ent.text.lower())
        
        enriched_keywords = local_keywords.copy()
        if use_web_search:
            query = f"software domain for project: {project[:100]}"
            search_keywords = web_search_extract_keywords(query, top_n=5)
            enriched_keywords.update(search_keywords)
        
        # Match and confirm
        domain_scores = {}
        for domain in COMMON_DOMAINS:
            confirmed, conf = confirm_domain_match(enriched_keywords, domain)
            if confirmed:
                domain_scores[domain] = conf
        
        top_domains = [d for d, _ in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]]
        overall_confidence = sum(domain_scores.values()) / max(1, len(domain_scores)) if domain_scores else 0.0
        
        domains_per_project[project_key] = {
            "domains": top_domains,
            "confirmed": bool(top_domains),
            "confidence": round(overall_confidence, 2)
        }
    
    if not domains_per_project:
        domains_per_project["no_projects"] = {"domains": [], "confirmed": False, "confidence": 0.0}
    
    return domains_per_project

def save_domain_keywords(query: str):
    keywords = web_search_extract_keywords(query)
    domain_keywords_col.update_one({"query": query}, {"$set": {"keywords": keywords}}, upsert=True)
    return keywords

def get_latest_domain_keywords(query: str) -> List[str]:
    record = domain_keywords_col.find_one({"query": query})
    if record:
        return record["keywords"]
    return save_domain_keywords(query)

def tokenize_text(text: str) -> List[str]:
    tokens = scoring.tokenize(text)
    tokens.extend([s for s in SKILLS_VOCAB if s in text.lower()])
    return list(set(tokens))

def score_resume_with_dynamic_keywords(resume: Dict, keywords: List[str]) -> Dict:
    jd_text = " ".join(keywords)
    # Call scoring's score_resume and add domains
    base_score = scoring.score_resume(resume, jd_text)
    base_score["project_domains"] = infer_project_domains(resume.get("projects", []))
    return base_score

@app.post("/upload/")
async def upload_resume(file: UploadFile = File(...), job_domain_query: str = "", background_tasks: BackgroundTasks = None):
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    parsed_data = parse_file(file_path, SKILLS_JSON_PATH)

    if job_domain_query:
        background_tasks.add_task(save_domain_keywords, job_domain_query)
        domain_keywords = get_latest_domain_keywords(job_domain_query)
    else:
        domain_keywords = []

    score_result = score_resume_with_dynamic_keywords(parsed_data, domain_keywords)

    # NEW: Compute project domains (already in score_result)
    project_domains = score_result.get("project_domains", {})

    resume_doc = {
        "filename": file.filename,
        "filepath": str(file_path),
        "parsed_data": parsed_data,
        "domain_query": job_domain_query,
        "domain_keywords": domain_keywords,
        "score": score_result.get("score"),
        "matched_keywords": score_result.get("matched"),
        "missing_keywords": score_result.get("missing"),
        "project_domains": project_domains,
    }
    inserted = resumes_col.insert_one(resume_doc)
    return {"id": str(inserted.inserted_id), "score": score_result.get("score")}

@app.get("/resume/{resume_id}")
def get_resume(resume_id: str):
    try:
        obj_id = ObjectId(resume_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid resume ID")
    resume = resumes_col.find_one({"_id": obj_id})
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    resume["_id"] = str(resume["_id"])
    return JSONResponse(content=resume)

@app.get("/resumes/")
def list_resumes():
    resumes = list(resumes_col.find({}, {"filename": 1, "score": 1}))
    for r in resumes:
        r["_id"] = str(r["_id"])
    return resumes

@app.post("/update_keywords/")
def update_keywords(query: str):
    keywords = save_domain_keywords(query)
    return {"query": query, "keywords": keywords}