import logging
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
logging.basicConfig(level=logging.INFO)

def choose_relevant_section(query, candidate_sections):
    """Ask LLM to pick the best section heading only (not the content)."""
    options = "\n".join(f"- {doc.metadata.get('section','Unknown')}" for doc in candidate_sections)

    prompt = f"""
    You are a section classifier. 
    The user asked: "{query}"

    Candidate SDS sections are:
    {options}

    Reply ONLY with the single most relevant section heading (exactly as written above).
    If none are relevant, reply "NONE".
    """
    resp = llm.invoke(prompt)
    return resp.content.strip()
