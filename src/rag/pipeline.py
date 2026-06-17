# src/rag/pipeline.py
"""Retrieval‑augmented generation (RAG) core for MedVoiceRAG.

This module loads the persistent ChromaDB collection created by
`src/ingest/build_index.py`, builds a retriever that returns the top‑5 most
relevant chunks for a given query, and defines `answer_question` which:

1. Retrieves the relevant chunks.
2. Calls OpenAI's ``gpt-4o-mini`` model with a prompt that forces the model to
   answer *only* using the retrieved context.
3. Citations are expressed as ``[PMID: <id>]``.
4. Returns the answer **and** a list of source documents (PMID, title, journal,
   year).

The script loads the OpenAI API key from a ``.env`` file (via ``python‑dotenv``)
and includes a simple ``__main__`` demo.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Import handling – support multiple LangChain versions / packages
# ---------------------------------------------------------------------------
try:
    # Preferred modern imports
    from langchain_chroma import Chroma
except ImportError:
    try:
        # Fallback to community package (deprecated but still works)
        from langchain_community.vectorstores import Chroma
    except ImportError:
        # Legacy import – works with older LangChain releases
        from langchain.vectorstores import Chroma

try:
    # Modern OpenAI chat model wrapper
    from langchain_openai import ChatOpenAI
except ImportError:
    # Older wrapper name
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.llms import OpenAI
    # Define a tiny shim so later code can use ``ChatOpenAI``
    class ChatOpenAI:
        def __init__(self, **kwargs):
            # ``model`` argument is ignored for the shim – we fall back to the
            # legacy ``OpenAI`` LLM which also respects ``OPENAI_API_KEY``.
            self.llm = OpenAI(**kwargs)
        def invoke(self, messages):
            # ``messages`` is a list of dicts with ``role`` and ``content``.
            # Concatenate them into a single prompt for the legacy LLM.
            prompt = "\n".join(m["content"] for m in messages)
            return {"content": self.llm(prompt)}

# ---------------------------------------------------------------------------
# Configuration (editable)
# ---------------------------------------------------------------------------
CHROMA_DIR = Path(__file__).parents[2] / "chroma_db"
CHROMA_COLLECTION_NAME = "pubmed"
TOP_K = 5  # number of chunks to retrieve per query

# Load environment variables (expects ``OPENAI_API_KEY`` in .env)
load_dotenv()

# ---------------------------------------------------------------------------
# Helper: build the persistent vector store and a retriever
# ---------------------------------------------------------------------------
def _load_vectorstore() -> Chroma:
    """Instantiate the persistent Chroma vector store.

    The collection was created by ``build_index.py`` and persisted to
    ``CHROMA_DIR``. ``Chroma`` will automatically load the existing collection.
    """
    # The embedding function is required even when loading an existing store.
    # We reuse the same OpenAI embedding model used during indexing.
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        # Fallback for older versions
        from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    return vectordb

# Load once at import time – cheap because the store is persisted on disk.
_vectorstore = _load_vectorstore()
_retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ---------------------------------------------------------------------------
# Core RAG function
# ---------------------------------------------------------------------------
def answer_question(query: str) -> Tuple[str, List[Dict]]:
    """Answer a user query using retrieved PubMed chunks.

    Parameters
    ----------
    query: str
        The natural‑language question.

    Returns
    -------
    answer: str
        The model's answer, with citations in the form ``[PMID: 12345678]``.
    sources: List[Dict]
        A list of dictionaries for each source document used, containing:
        ``pmid``, ``title``, ``journal``, ``year``.
    """
    # Retrieve the most relevant chunks using the retriever's invoke method (compatible across LangChain versions).
    docs = _retriever.invoke(query)

    # Build a concise context string – each chunk is separated by a line.
    context_parts = []
    source_meta = []
    for doc in docs:
        meta = doc.metadata
        pmid = meta.get("pmid", "N/A")
        title = meta.get("title", "")
        journal = meta.get("journal", "")
        year = meta.get("year", "")
        # Keep a record of the source metadata for the return value.
        source_meta.append({"pmid": pmid, "title": title, "journal": journal, "year": year})
        # Append the chunk with an explicit citation marker.
        context_parts.append(f"[PMID: {pmid}] {doc.page_content}")
    context = "\n\n".join(context_parts)

    # System prompt that forces citation style.
    system_prompt = (
        "You are an expert medical assistant. Answer the user's question "
        "using **only** the information provided in the context below. "
        "If you cite information, reference the PMID exactly as ``[PMID: xxxx]``. "
        "Do not fabricate sources or add information that is not present in the context."
    )

    # Construct the message list for the LLM.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    # Initialise the chat model.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)
    # The response may be a dict (modern API) or have ``content`` attribute.
    if isinstance(response, dict):
        answer = response.get("content", "")
    else:
        # Fallback for older shim implementation.
        answer = getattr(response, "content", "")

    return answer.strip(), source_meta

# ---------------------------------------------------------------------------
# Demo execution block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_question = "What is the role of natalizumab in multiple sclerosis?"
    ans, sources = answer_question(sample_question)
    print("--- Answer ---")
    print(ans)
    print("\n--- Sources ---")
    for src in sources:
        print(f"[PMID: {src['pmid']}] {src['title']} ({src['journal']}, {src['year']})")
