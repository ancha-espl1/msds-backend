import os, hashlib
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import logging
from typing import List

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _corpus_signature(data_path: str) -> str:
    """Hash of filenames + sizes + mtimes to avoid re-embedding unchanged corpora."""
    acc = hashlib.sha256()
    for root, _, files in os.walk(data_path):
        for f in sorted(files):
            if not f.lower().endswith(".pdf"):
                continue
            fp = os.path.join(root, f)
            st = os.stat(fp)
            acc.update(f.encode("utf-8"))
            acc.update(str(st.st_size).encode("utf-8"))
            acc.update(str(int(st.st_mtime)).encode("utf-8"))
    return acc.hexdigest()

def make_vectordb_and_retriever(
    documents: List[Document],
    persist_directory: str,
    data_path: str,
    collection_name: str = "sds",
    k: int = 5,
):
    embeddings = OpenAIEmbeddings()

    sig = _corpus_signature(data_path)
    sig_file = os.path.join(persist_directory, ".corpus.sig")

    os.makedirs(persist_directory, exist_ok=True)

    need_rebuild = True
    if os.path.exists(sig_file):
        with open(sig_file, "r", encoding="utf-8") as f:
            prev = f.read().strip()
        if prev == sig:
            need_rebuild = False

    if need_rebuild:
        # Fresh build
        logger.info("Building Chroma from %d documents into %s ...", len(documents), persist_directory)
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        with open(sig_file, "w", encoding="utf-8") as f:
            f.write(sig)
        logger.info("Chroma built. Persisted to %s", persist_directory)
    else:
        # Load existing
        logger.info("Loading existing Chroma from %s", persist_directory)
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return vectordb, retriever