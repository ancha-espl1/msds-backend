import os
import re
from langchain_core.documents import Document
from typing import List, Dict

# ---------------------------
# 1. Extract Product Name from SDS filename
# ---------------------------
def extract_product_name(file_path: str) -> str:
    """Extract product name from SDS file naming convention."""
    # Example filename: ESPL-Aurion-Nld-En-140225-5.txt
    name = os.path.basename(file_path)
    name = name.replace("ESPL-", "").replace(".txt", "")
    return name.split("-")[0].strip()

# ---------------------------
# 2. Apply File-Level Metadata
# ---------------------------
def apply_file_level_metadata(file_path: str, raw_text: str) -> List[Document]:
    product_name = extract_product_name(file_path)
    sections = re.split(r"(?=Section\s+\d+[:.])", raw_text)
    documents = []

    for sec in sections:
        sec_clean = sec.strip()
        if not sec_clean:
            continue

        match = re.match(r"(Section\s+\d+[:.])", sec_clean)
        section_title = match.group(1) if match else "UNKNOWN"

        documents.append(
            Document(
                page_content=sec_clean,
                metadata={
                    "source": file_path,
                    "product_name": product_name,
                    "section": section_title
                }
            )
        )
    return documents

# ---------------------------
# 3. Split Pages & Identify Headers/Footers
# ---------------------------
def split_pages(raw_text: str) -> List[str]:
    """Split text into pages based on form feed or SDS page numbers."""
    return re.split(r"\f|\n\s*Page\s+\d+\s+of\s+\d+", raw_text)

def identify_repeated_headers_footers(pages: List[str]) -> Dict[str, int]:
    """Identify candidate header/footer lines occurring on most pages."""
    line_frequency = {}
    for page in pages:
        for line in page.splitlines():
            line = line.strip()
            if len(line) < 5:
                continue
            line_frequency[line] = line_frequency.get(line, 0) + 1

    # Consider a header/footer if it appears in >70% of pages
    threshold = max(2, int(len(pages) * 0.7))
    return {line: count for line, count in line_frequency.items() if count >= threshold}

def remove_headers_footers(pages: List[str], repeated_lines: Dict[str, int]) -> str:
    """Remove repeated headers/footers from pages."""
    clean_pages = []
    for page in pages:
        clean_lines = [ln for ln in page.splitlines() if ln.strip() not in repeated_lines]
        clean_pages.append("\n".join(clean_lines))
    return "\n".join(clean_pages)

# ---------------------------
# 4. Loader Function
# ---------------------------
def load_sds_documents(folder_path: str = "data/sds_text/") -> List[Document]:
    """Load, clean, and prepare SDS documents."""
    docs = []
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        # Step 1: Split pages and remove headers/footers
        pages = split_pages(raw_text)
        repeated_lines = identify_repeated_headers_footers(pages)
        clean_text = remove_headers_footers(pages, repeated_lines)

        # Step 2: Create Documents with metadata
        file_docs = apply_file_level_metadata(file_path, clean_text)
        docs.extend(file_docs)

    return docs
