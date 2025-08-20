from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

RAG_TEMPLATE = """
You are a strict SDS assistant that helps answer safety-related questions using only the official SDS text.

Only answer using the content from the SDS document chunks below. Do not paraphrase, summarize, or add any new information. Copy the relevant lines exactly from the document.

SDS Content:
{context}

Question: {question}

Answer:"""

def get_rag_chain(retriever):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    prompt = PromptTemplate(input_variables=["context", "question"], template=RAG_TEMPLATE)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
