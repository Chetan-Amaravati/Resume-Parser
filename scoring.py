
import re
from typing import List, Dict
import pandas as pd

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z+#\.]{2,}", text.lower())

def keywords_from_jd(jd: str) -> List[str]:
    toks = tokenize(jd)
    # remove generic words
    stop = set(["and","or","with","the","a","an","to","of","in","on","for","using","experience","skills","skill","developer","engineer","analyst"])
    return [t for t in toks if t not in stop]

def score_resume(resume: Dict, jd: str) -> Dict:
    jd_keys = set(keywords_from_jd(jd))
    res_keys = set([k.lower() for k in resume.get("skills", [])])
    # also include tokens from projects/experience text
    extra_text = " ".join(resume.get("projects", []) + [b for e in resume.get("experience", []) for b in e.get("bullets", [])])
    res_tokens = set(tokenize(extra_text))
    matched = jd_keys & (res_keys | res_tokens)
    coverage = len(matched) / max(1, len(jd_keys))
    score = int(round(coverage * 100))
    missing = sorted(list(jd_keys - (res_keys | res_tokens)))
    return {"score": score, "matched": sorted(list(matched)), "missing": missing}

def score_dataframe(df: pd.DataFrame, jd: str) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        if isinstance(row.get("error"), str):
            out.append({"file": row["file"], "score": 0, "missing": [], "matched": [], "error": row["error"]})
            continue
        s = score_resume(row.to_dict(), jd)
        out.append({"file": row["file"], **s})
    return pd.DataFrame(out).sort_values("score", ascending=False).reset_index(drop=True)

# Simple offline "summary" using extractive TextRank-like scoring
def summarize(text: str, max_sentences: int = 3) -> str:
    # naive split
    sents = re.split(r"(?<=[.?!])\s+", text.strip())
    if len(sents) <= max_sentences:
        return " ".join(sents)
    # score sentences by keyword frequency
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
