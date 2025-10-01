import os
from typing import Tuple, Dict

import pdfplumber
from docx import Document as DocxDocument


def parse_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    
    return "\n\n".join(text)


def parse_docx(path: str) -> str:
    """Extract text from a DOCX file."""
    doc = DocxDocument(path)
    paragraphs = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
    return "\n\n".join(paragraphs)


def parse_txt(path: str) -> str:
    """Extract text from a TXT file."""
    with open(path, 'r', encoding='utf-8', errors="ignore") as file:
        return file.read()


def parse_file(path: str) -> Tuple[str, Dict]:
    """Parse a file and return its text content along with metadata."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.pdf':
        text = parse_pdf(path)
    elif ext == '.docx':
        text = parse_docx(path)
    elif ext in ['.txt', '.md', '.csv']:
        text = parse_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    metadata = {
        "source": os.path.basename(path),
        "file_type": ext.lstrip('.'),
        "path": path,
        "size_bytes": os.path.getsize(path)
    }
    
    return text, metadata
