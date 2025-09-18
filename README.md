
# Offline Resume Parser — 7th Semester Project

**Works 100% offline**. Parses PDF/DOCX/TXT, extracts sections (Skills, Education, Projects, Experience, Contact), computes **Section Confidence**, performs **Resume Scoring** against a JD, shows **Skill Gap**, and generates an **offline extractive summary**. Includes a **synthetic dataset of 15 resumes**.

## Features
- PDF/DOCX/TXT parsing (PyMuPDF, python-docx).
- Robust regex/NLP-lite extraction.
- Section Confidence % per field.
- Resume Scoring out of 100 + Missing Skills list.
- Offline extractive summary (TextRank-like).
- Multi-language header support: English + basic Kannada variants.
- Streamlit UI with upload, dataset toggle, tables, and visuals.
- Word Cloud instructions for skills visualization (optional section).

## Project Structure
```
resume_parser_project/
├─ dataset/            # 15 resumes (7 PDFs, 5 DOCX, 3 TXT) incl. your two PDFs
├─ app.py              # Streamlit app
├─ parser.py           # Parsing logic
├─ scoring.py          # Scoring + offline summary
├─ skills.json         # Skills dictionary
├─ screenshots/        # Example images
├─ requirements.txt    # Dependencies
└─ README.md
```

## Installation (Windows, Offline)
> **Note:** Ensure you have Python 3.10+ and a working C++ build tools if needed by PyMuPDF.

1. Create and activate a virtual environment:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies from `requirements.txt` (these wheels can be copied to your machine if completely offline):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## How to Use
1. Launch the app.
2. (Optional) Upload resumes or use the included dataset (toggle in sidebar).
3. Paste a Job Description (JD).
4. Click **Parse Resume(s)**, then **Score & Skill Gap**.
5. Explore confidence, matched & missing keywords, and sample summary.

## Word Cloud (Optional)
To visualize common skills:
1. Install `wordcloud` (already in requirements).
2. In the app, you can add a small block to create a word cloud from extracted skills.
   (Left minimal in `app.py` to keep UI clean, but the code snippet is in comments below.)

## Screenshots
The `screenshots/` folder has example output images you can paste into your report.

## Notes
- The offline summary is extractive (keyword-frequency based) to avoid heavy model downloads.
- Kannada support covers header detection; detailed NLP for all Indic languages is out-of-scope but this showcases multilingual readiness.
- Extend `skills.json` for domain-specific parsing.
