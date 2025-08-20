import os, json, hashlib, glob
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

INDEX_META = "index_meta.json"
INDEX_VERSION = "sds-index-v2"  # bump when loader changes

def _hash_corpus(data_path="data/sds_text/"):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(data_path, "*.*"))):
        h.update(os.path.basename(p).encode())
        h.update(str(os.path.getmtime(p)).encode())
    return h.hexdigest()

def make_vectordb_and_retriever(documents=None, persist_directory="chroma_store", data_path="data/sds_text/"):
    embeddings = OpenAIEmbeddings()
    need_rebuild = True
    os.makedirs(persist_directory, exist_ok=True)
    meta_path = os.path.join(persist_directory, INDEX_META)
    corpus_hash = _hash_corpus(data_path)

    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, "r"))
        if meta.get("version") == INDEX_VERSION and meta.get("corpus_hash") == corpus_hash:
            need_rebuild = False

    if not need_rebuild and os.listdir(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="sds")
        print(f"Loaded existing Chroma DB from {persist_directory}")
    else:
        if not documents:
            raise ValueError("No documents provided to create a new vector store.")
        vectordb = Chroma.from_documents(
            documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="sds"
        )
        vectordb.persist()
        json.dump({"version": INDEX_VERSION, "corpus_hash": corpus_hash}, open(meta_path, "w"))
        print(f"Created new Chroma DB at {persist_directory}")

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return vectordb, retriever
