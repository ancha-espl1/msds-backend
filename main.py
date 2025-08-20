from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.loader import load_sds_documents
from app.retriever import make_vectordb_and_retriever as create_retriever
from app.qa_engine import get_rag_chain
import logging
from app.section_picker import choose_relevant_section

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Allow React frontend origin
origins = [
    "http://localhost:3001",  # React dev server
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

docs = load_sds_documents()
vectordb, retriever = create_retriever(docs)

def extract_section_verbatim(docs, target_heading):
    for doc in docs:
        if doc.metadata.get("section") == target_heading:
            return doc.page_content, doc.metadata
    return None, None

@app.post("/query")
def query_sds(req: QueryRequest):
    query = req.question
    candidates = retriever.get_relevant_documents(query)
    logging.info("Retrieved %d candidates for query=%s", len(candidates), query)

    for i, doc in enumerate(candidates, start=1):
        logging.info(
            "Candidate %d: section=%s, source=%s",
            i,
            doc.metadata.get("section", "UNKNOWN"),
            doc.metadata.get("source", "UNKNOWN"),
    )

    if not candidates:
        return {"answer": "ANSWER NOT FOUND IN SDS", "source": None}

    # Let LLM pick section heading
    section = choose_relevant_section(query, candidates)
    logging.info("LLM picked section=%s", section)

    if section != "NONE":
        content, meta = extract_section_verbatim(candidates, section)
        if content:
            return {"answer": content, "source": meta}

    # fallback = return top doc verbatim
    top = candidates[0]
    logging.warning("No section match found; returning top doc verbatim source=%s", top.metadata.get("source"))
    return {"answer": top.page_content, "source": top.metadata}