import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.loader_pdf import load_sds_documents
from app.retriever import make_vectordb_and_retriever
import logging
from app.section_picker import choose_relevant_section
import logging
from openai import OpenAI

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Allow React frontend origin
origins = [
    "http://localhost:3000",  # React dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)

class QueryRequest(BaseModel):
    question: str

# ---- Build/load vector store from PDFs ----
PDF_DIR = "data/sds_pdf"
PERSIST_DIR = "chroma_store_pdf"

docs = load_sds_documents(pdf_dir=PDF_DIR)   # returns List[Document]
vectordb, retriever = make_vectordb_and_retriever(
    documents=docs,
    persist_directory=PERSIST_DIR,
    data_path=PDF_DIR,
    collection_name="sds"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PRODUCT_SYNONYMS = {
    "AURION CWFS Gelatin": ["CWFS Gelatin", "Aurion Gelatin 40%", "AURION CWFS Gelatin (40%)"],
    "India Ink Control": ["India Ink", "Ink Control Sample"],
    "MOBIL SUPER ALL-IN-ONE PROTECTION 0W-20": ["MOBIL SUPER ALL-IN-ONE PROTECTION", "MOBIL SUPER 0W-20", "MOBIL SUPER"],
    "Mobiltrans HD 30 Dyed Blue": ["Mobiltrans HD 30", "Mobiltrans HD 30"],
    "FLEXGRIT": ["FLEXGRIT"],
    "pH Buffer 10.01 colourless": ["pH Buffer 10.01", "pH Buffer colourless"],
    "SAFTIGRIT BLUE (PREMIUM)": ["SAFTIGRIT BLUE", "SAFTIGRIT BLUE PREMIUM"],
}

def extract_product_hint(query: str):
    for product, synonyms in PRODUCT_SYNONYMS.items():
        if product.lower() in query.lower():
            return product
        for syn in synonyms:
            if syn.lower() in query.lower():
                return product
    return None

def extract_product_hint_llm(query: str):
    """
    Ask the LLM to extract a probable product name from the query.
    """
    try:
        prompt = f"""
        Extract the exact SDS product name from the user query below.
        Query: "{query}"
        Return only the product name or return "NONE" if unsure.
        """
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error("LLM product extraction failed: %s", str(e))
        return None

@app.post("/query")
def query_sds(req: QueryRequest):
    query = req.question
    # Pass 1: Try synonym-based hinting
    product_hint = extract_product_hint(query)
    logging.info("Extracted product hint: %s", product_hint)
    # Pass 2: Fallback to LLM extraction
    if not product_hint:
        product_hint = extract_product_hint_llm(query)
        if product_hint and product_hint.upper() == "NONE":
            product_hint = None

    if product_hint:
        logging.info("Using product hint: %s", product_hint)
        candidates = retriever.vectorstore.similarity_search(
                query,
                k=5,
                filter={"product_name": product_hint} if product_hint else None
        )
        if not candidates:  # fallback if product filter too strict
            logging.warning("No hits with product filter, falling back to full search.")
            candidates = retriever.invoke(query)
    else:
        # ❓ OPTION 1: Allow fallback without filter
        candidates = retriever.invoke(query)
        # ❓ OPTION 2: Force user to refine question instead
        # return {"answer": "I couldn’t identify the product from your question. Please mention the product name."}

    if not candidates:
        return {"answer": "ANSWER NOT FOUND IN SDS", "source": None}

    logging.info("Retrieved %d candidates for query=%s", len(candidates), query)
    
    for i, doc in enumerate(candidates, start=1):
        logging.info(
            "Candidate %d: section=%s, product=%s, source=%s",
            i,
            doc.metadata.get("section", "UNKNOWN"),
            doc.metadata.get("product", "UNKNOWN"),
            doc.metadata.get("source", "UNKNOWN"),
        )
        
    section = choose_relevant_section(query, candidates)
    logger.info("LLM picked section=%s", section)

    # Return the exact chunk matching that section, verbatim
    for doc in candidates:
        if doc.metadata.get("section") == section:
            return {"answer": doc.page_content, "source": doc.metadata}

    # Fallback: top chunk
    return {"answer": candidates[0].page_content, "source": candidates[0].metadata}