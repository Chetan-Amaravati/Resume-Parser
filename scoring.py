import re
from typing import List, Dict
import pandas as pd
import requests
from googlesearch import search
import spacy
from typing import List, Dict, Tuple

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Common domains
COMMON_DOMAINS = [
    "web development", "mobile development", "machine learning", "data science", "artificial intelligence",
    "blockchain", "cybersecurity", "embedded systems", "cloud computing", "devops", "game development",
    "iot", "big data", "software engineering", "ui/ux design", "database management", "networking",
    "computer vision", "deep learning", "document management", "decentralized systems", "ride sharing",
    "natural language processing", "resume parsing", "notarization systems"
]

def web_search_extract_keywords(query: str, top_n: int = 3) -> List[str]:
    """Extract keywords from Google search results using googlesearch-python"""
    keywords = set()
    try:
        results = list(search(query, num_results=top_n, lang="en"))
        text_blob = " ".join([result.title + " " + result.description for result in results if hasattr(result, 'title') and result.title])
        
        if not text_blob:
            print(f"No search results for query: {query}")
            return []
        
        if nlp:
            doc = nlp(text_blob)
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 2 and not chunk.text.lower() in {'the', 'and', 'for', 'with', 'using'}:
                    keywords.add(chunk.text.lower())
            for ent in doc.ents:
                if ent.label_ in {'ORG', 'PRODUCT', 'EVENT'}:
                    keywords.add(ent.text.lower())
        else:
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text_blob.lower())
            keywords = set(tokens[:20])
        
        print(f"Search for '{query}': Extracted {len(keywords)} keywords (e.g., {list(keywords)[:3]})")
        return list(keywords)
    except Exception as e:
        print(f"Web search failed for '{query}': {e}. Falling back to local keywords.")
        return []

def confirm_domain_match(project_keywords: set, domain: str) -> Tuple[bool, float]:
    """Confirm if project relates to domain via keyword overlap"""
    domain_toks = set(re.findall(r'\b[a-zA-Z]{3,}\b', domain.lower()))
    overlap = len(project_keywords & domain_toks)
    confidence = overlap / max(1, len(domain_toks))
    confirmed = confidence > 0.3
    return confirmed, confidence

def infer_project_domains(projects: List[str], use_web_search: bool = True) -> Dict[str, Dict[str, any]]:
    """
    Infer and confirm domains for projects using web search.
    Returns: {project_key: {"domains": [top_domains], "confirmed": bool, "confidence": float}}
    """
    domains_per_project = {}
    
    for project in projects:
        if not project.strip():
            continue
        
        project_key = project[:50] + "..."
        
        # Local keywords from project text
        local_keywords = set()
        if nlp:
            doc = nlp(project)
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 2:
                    local_keywords.add(chunk.text.lower())
            for ent in doc.ents:
                local_keywords.add(ent.text.lower())
        local_keywords.update(set(tokenize(project)))
        
        # Enrich with web search
        enriched_keywords = local_keywords.copy()
        if use_web_search:
            query_terms = [t for t in re.findall(r'\b[a-zA-Z]{3,}\b', project.lower()) if t not in {'for', 'and', 'the', 'a', 'an'}][:5]
            query = f"what software domain or technology stack for project: {project[:120]} {' '.join(query_terms)} examples"
            search_keywords = web_search_extract_keywords(query, top_n=5)
            enriched_keywords.update(search_keywords)
            print(f"Web search for '{project[:30]}...': Query='{query}', Enriched with {len(search_keywords)} keywords")
        
        # Match and confirm domains
        domain_scores = {}
        for domain in COMMON_DOMAINS:
            confirmed, conf = confirm_domain_match(enriched_keywords, domain)
            if confirmed:
                domain_scores[domain] = conf
        
        # Top confirmed domains
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

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z+#\.]{2,}", text.lower())

def keywords_from_jd(jd: str) -> List[str]:
    toks = tokenize(jd)
    stop = set(["and","or","with","the","a","an","to","of","in","on","for","using","experience","skills","skill","developer","engineer","analyst"])
    return [t for t in toks if t not in stop]

def score_resume(resume: Dict, jd: str) -> Dict:
    # Safely handle skills, converting non-iterable types to empty list
    skills_value = resume.get("skills", [])
    if not isinstance(skills_value, (list, tuple)):
        print(f"Debug: Invalid skills type {type(skills_value)} with value {skills_value} for resume {resume.get('file', 'unknown')}")
        skills_value = []
    res_keys = set([k.lower() for k in skills_value if isinstance(k, str)])
    
    jd_keys = set(keywords_from_jd(jd))
    extra_text = " ".join(resume.get("projects", []) + [b for e in resume.get("experience", []) for b in e.get("bullets", [])])
    res_tokens = set(tokenize(extra_text))
    matched = jd_keys & (res_keys | res_tokens)
    coverage = len(matched) / max(1, len(jd_keys))
    score = int(round(coverage * 100))
    missing = sorted(list(jd_keys - (res_keys | res_tokens)))
    
    project_domains = infer_project_domains(resume.get("projects", []))
    
    return {
        "score": score,
        "matched": sorted(list(matched)),
        "missing": missing,
        "project_domains": project_domains
    }

def score_dataframe(df: pd.DataFrame, jd: str) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        if isinstance(row.get("error"), str):
            out.append({"file": row["file"], "score": 0, "missing": [], "matched": [], "error": row["error"]})
            continue
        s = score_resume(row.to_dict(), jd)
        out.append({"file": row["file"], **s})
    return pd.DataFrame(out).sort_values("score", ascending=False).reset_index(drop=True)

def summarize(text: str, max_sentences: int = 3) -> str:
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    sents = re.split(r"(?<=[.?!])\s+", text)
    if len(sents) <= max_sentences:
        return " ".join(sents)
    toks = tokenize(text)
    if not toks:
        return " ".join(sents[:max_sentences])
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    scores = []
    for i, s in enumerate(sents):
        stoks = tokenize(s)
        sc = sum(freq.get(t, 0) for t in stoks) / max(1, len(stoks))
        scores.append((sc, i, s))
    top = [s for _,_,s in sorted(scores, reverse=True)[:max_sentences]]
    return " ".join(top)