import os
import re
from typing import List, Tuple
import fitz  # PyMuPDF
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = "chroma_store"
PDF_DIR = "data/sds_pdf"

# --------------------------
# PDF reading (page-wise) + optional boilerplate filtering
# --------------------------
def _read_pdf_pages(pdf_path: str) -> List[str]:
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            txt = page.get_text("text") or ""
            pages.append(txt)
    return pages

def _find_repeated_lines(pages: List[str], threshold: float = 0.6) -> set:
    """Identify header/footer lines repeated on >= threshold of pages."""
    if not pages:
        return set()
    freq = {}
    for p in pages:
        for line in p.splitlines():
            ln = re.sub(r"\s+", " ", line).strip()
            if ln:
                freq[ln] = freq.get(ln, 0) + 1
    cutoff = max(2, int(len(pages) * threshold))
    repeated = {line for line, c in freq.items() if c >= cutoff}

    # Add common boilerplate patterns (conservative)
    common = [
        r"^Page\s+\d+\s+of\s+\d+",
        r"^Revision\s+date",
        r"^Date\s+of\s+print",
        r"^SAFETY\s+DATA\s+SHEET",
        r"according to Regulation",
        r"^Document number",
    ]
    for s in list(freq.keys()):
        for pat in common:
            if re.search(pat, s, re.IGNORECASE):
                repeated.add(s)
                break
    return repeated

def _strip_repeated_lines(pages: List[str], repeated: set) -> str:
    cleaned_pages = []
    for p in pages:
        kept = []
        for line in p.splitlines():
            ln = re.sub(r"\s+", " ", line).strip()
            if ln and ln not in repeated:
                kept.append(line)
        cleaned_pages.append("\n".join(kept))
    return "\n".join(cleaned_pages)

def _read_pdf_text_clean(pdf_path: str) -> str:
    pages = _read_pdf_pages(pdf_path)
    if not pages:
        return ""
    repeated = _find_repeated_lines(pages, threshold=0.6)
    full_text = _strip_repeated_lines(pages, repeated)
    return full_text.strip()


# --------------------------
# Section 1 extraction
# --------------------------
_SECTION1_BLOCK_PATTERNS = [
    # "Section 1: ..." up to next section 2 marker
    r"(?is)^\s*Section\s*0*1\b.*?(?=^\s*Section\s*0*2\b|^\s*2[\.\-]?\s+|^\s*2\s+[A-Z]|$\Z)",
    r"(?is)^\s*SECTION\s*0*1\b.*?(?=^\s*SECTION\s*0*2\b|^\s*2[\.\-]?\s+|^\s*2\s+[A-Z]|$\Z)",
    # "1. Identification" (no word 'Section'), up to "2."
    r"(?is)^\s*1[\.\-]?\s+[A-Z][^\n]*\n.*?(?=^\s*2[\.\-]?\s+|^\s*2\s+[A-Z]|^\s*Section\s*0*2\b|^\s*SECTION\s*0*2\b|$\Z)",
    # "1 IDENTIFICATION" (all caps, no punctuation)
    r"(?is)^\s*1\s+[A-Z][A-Z \-]{3,}\n.*?(?=^\s*2\s+[A-Z]|^\s*2[\.\-]?\s+|^\s*Section\s*0*2\b|^\s*SECTION\s*0*2\b|$\Z)",
]

def _extract_section1_block(full_text: str) -> str:
    for pat in _SECTION1_BLOCK_PATTERNS:
        m = re.search(pat, full_text, re.MULTILINE)
        if m:
            s1 = m.group(0).strip()
            logger.debug("Section 1 block found with pattern: %s", pat)
            return s1
    # Fallback: first ~2500 chars
    logger.warning("Section 1 block not found; using first 2500 chars as fallback.")
    return full_text[:2500]

# --------------------------
# Product name extraction (robust)
# --------------------------
# Labels we accept for product name (order matters; higher priority first)
_NAME_LABELS = [
    "Product Name",
    "Product",
    "Trade Name",
    "Product Identifier",
    "Chemical Name",
    "Material Name",
    "Substance Name",
    "Name",  # lowest priority; can conflict with "Supplier Name"
]
# Labels we explicitly ignore
_IGNORE_LABELS = [
    "Product Use", "Intended Use", "Recommended use",
    "Supplier", "Manufacturer", "Company", "Emergency", "Address", "Phone",
]

def _is_ignored_label(label: str) -> bool:
    lab = label.strip().lower()
    return any(lab.startswith(ig.lower()) for ig in _IGNORE_LABELS)

def _clean_value(val: str) -> str:
    v = re.sub(r"\s+", " ", val).strip()
    # Trim super-noisy tail tokens
    v = re.sub(r"(?:SDS\s*No\.?|CAS\s*No\.?).*$", "", v, flags=re.IGNORECASE).strip()
    return v

def extract_product_name(full_text: str, file_name: str) -> str:
    s1 = _extract_section1_block(full_text)
    logger.debug("Section 1 text (first 600 chars): %s", s1[:600])

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s1.splitlines() if ln.strip()]
    # Scan lines for label:value or label on one line, value on next
    for i, ln in enumerate(lines):
        # Split on ":" if present; otherwise try label-only + lookahead
        if ":" in ln:
            left, right = [t.strip() for t in ln.split(":", 1)]
            if _is_ignored_label(left):
                continue
            # Match acceptable labels (with variations like "Trade Name")
            if any(re.fullmatch(fr"{lab}", left, flags=re.IGNORECASE) for lab in _NAME_LABELS):
                val = _clean_value(right)
                # Avoid lines like "Product: Use: Automotive Degreaser"
                if re.match(r"(?i)\b(use|intended)\b", val):
                    continue
                if val:
                    logger.info("Extracted product name (label:value) for %s: %s", file_name, val)
                    return val
        else:
            # If no colon, see if the line is a label and the next line is the value
            if any(re.fullmatch(fr"{lab}", ln, flags=re.IGNORECASE) for lab in _NAME_LABELS):
                if _is_ignored_label(ln):
                    continue
                # next non-empty line as value
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    val = _clean_value(lines[j])
                    # do not accept if next line looks like another label or an ignored label
                    if not any(re.fullmatch(fr"{lab}", val, flags=re.IGNORECASE) for lab in _NAME_LABELS) and not _is_ignored_label(val):
                        logger.info("Extracted product name (label\\nvalue) for %s: %s", file_name, val)
                        return val

    logger.warning("Product name NOT found in Section 1 for %s; returning UNKNOWN", file_name)
    return "UNKNOWN"

# --------------------------
# Section splitting (broad)
# --------------------------
_SECTION_SPLIT_LOOKAHEAD = re.compile(
    r"""(?imx)
    (?=^\s*
        (?:                                    # Accept any of these heading forms:
            (?:Section|SECTION|Sec|SEC)\s*0*\d+ # "Section 7" / "SECTION 07"
            (?:\s*[:\.\-\u2013]\s*|\s+)        # delimiter or whitespace
          | \d{1,2}[\.\-]\s+[A-Z]              # "1. IDENTIFICATION"
          | \d{1,2}\s+[A-Z][A-Z \-/]{3,}       # "1 IDENTIFICATION"
        )
    )
    """,
    re.MULTILINE,
)

def _split_into_sections(full_text: str) -> List[Tuple[str, str]]:
    parts = _SECTION_SPLIT_LOOKAHEAD.split(full_text)
    chunks: List[Tuple[str, str]] = []

    if len(parts) > 1:
        for part in parts:
            part = part.strip()
            if not part:
                continue
            first_line = part.splitlines()[0].strip()
            title = first_line
            chunks.append((title, part))
        logger.info("Detected %d sections via headings.", len(chunks))
        return chunks

    # Fallback: recursive splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ".", " "])
    chunks = [("FULL", ch) for ch in splitter.split_text(full_text)]
    logger.warning("No explicit sections found; using recursive chunks: %d", len(chunks))
    return chunks

# --------------------------
# Public API: return Documents (NOT a Chroma object)
# --------------------------
def load_sds_documents(pdf_dir: str = PDF_DIR) -> List[Document]:
    """
    Read PDFs, clean text, extract product_name, split into sections,
    and return a list of LangChain Documents. No DB writes here.
    """
    documents: List[Document] = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(pdf_dir, fname)
        logger.info("Processing PDF: %s", fname)

        text = _read_pdf_text_clean(fpath)
        if not text:
            logger.warning("No text found in %s; skipping.", fname)
            continue

        product_name = extract_product_name(text, fname)
        sections = _split_into_sections(text)

        for section_title, section_text in sections:
            documents.append(
                Document(
                    page_content=section_text,
                    metadata={
                        "source": fpath,           # full path is handy later
                        "file_name": fname,
                        "product_name": product_name,  # <<< consistent key
                        "section": section_title,
                        "format": "pdf",
                    },
                )
            )
        logger.info("Added %d chunks from %s (product_name=%s)", len(sections), fname, product_name)
    return documents