import os
import re
from langchain_core.documents import Document

SECTION_RE = re.compile(r'(?mi)^\s*Section\s+(\d+(?:\.\d+)?)\s*[:\-]\s*(.+?)\s*$', re.MULTILINE)
SUBSEC_RE  = re.compile(r'(?mi)^\s*(\d+\.\d+)\s*[:\-]?\s*(.+?)\s*$', re.MULTILINE)

def _normalize_ws(s: str) -> str:
    # keep newlines, collapse tabs & runs of spaces
    s = s.replace('\r', '')
    s = re.sub(r'[ \t]+', ' ', s)
    return s

def load_sds_documents(data_path="data/sds_text/"):
    docs = []
    for fname in sorted(os.listdir(data_path)):
        if not fname.lower().endswith(('.txt', '.pdf')):
            continue
        path = os.path.join(data_path, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()

        text = _normalize_ws(raw)

        # Split top-level sections
        # We find all matches and slice between them
        section_matches = list(SECTION_RE.finditer(text))
        if not section_matches:
            # fallback: single doc
            docs.append(Document(page_content=text, metadata={"source": fname, "section": "FULL"}))
            continue

        for i, m in enumerate(section_matches):
            sec_id = m.group(1)        # e.g., "7" or "7.1"
            sec_title = m.group(2)     # e.g., "Handling and Storage"
            start = m.end()
            end = section_matches[i+1].start() if i+1 < len(section_matches) else len(text)
            sec_block = text[start:end].strip()

            # Create main-section doc
            section_full = f"Section {sec_id}: {sec_title}".strip()
            docs.append(Document(
                page_content=f"{section_full}\n{sec_block}",
                metadata={"source": fname, "section": section_full, "section_id": sec_id, "section_title": sec_title}
            ))

            # Also split into subsections inside this block (e.g., 7.1 / 7.2 etc.)
            sub_matches = list(SUBSEC_RE.finditer(sec_block))
            for j, sm in enumerate(sub_matches):
                sub_id = sm.group(1)       # "7.1"
                sub_title = sm.group(2)    # "Precautions for safe handling"
                s_start = sm.end()
                s_end = sub_matches[j+1].start() if j+1 < len(sub_matches) else len(sec_block)
                sub_content = sec_block[s_start:s_end].strip()
                sub_full = f"Section {sub_id}: {sub_title}".strip()

                docs.append(Document(
                    page_content=f"{sub_full}\n{sub_content}",
                    metadata={
                        "source": fname,
                        "section": sub_full,
                        "section_id": sub_id,
                        "section_title": sub_title,
                        "parent_section": section_full
                    }
                ))
    return docs