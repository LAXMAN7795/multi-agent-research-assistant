"""
retrieval_tool.py — RAG Pipeline with ChromaDB
AI Research Agent
Handles: document ingestion, chunking, embedding, vector storage, and retrieval.
Embedding model : sentence-transformers/all-MiniLM-L6-v2 (runs locally, free)
Vector store    : ChromaDB (local persistent storage)
"""

import logging
import sys
import hashlib
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


# ── Embedding model (singleton — loaded once, reused everywhere) ──────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFace embedding model instance.
    Downloads the model on first call (~90 MB), then loads from cache.

    Returns:
        HuggingFaceEmbeddings instance using all-MiniLM-L6-v2.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name      = EMBEDDING_MODEL,
            model_kwargs    = {"device": "cpu"},
            encode_kwargs   = {"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
    return _embeddings


# ── Vector store (singleton) ──────────────────────────────────────────────────
_vectorstore: Optional[Chroma] = None

def get_vectorstore() -> Chroma:
    """
    Returns a cached ChromaDB vectorstore instance.
    Creates the collection if it does not already exist.

    Returns:
        Chroma vectorstore connected to the persistent local DB.
    """
    global _vectorstore
    if _vectorstore is None:
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        logger.info("Connecting to ChromaDB at: %s", CHROMA_DB_PATH)
        _vectorstore = Chroma(
            collection_name     = CHROMA_COLLECTION,
            embedding_function  = get_embeddings(),
            persist_directory   = CHROMA_DB_PATH,
        )
        count = _vectorstore._collection.count()
        logger.info("ChromaDB ready | collection: '%s' | documents: %d", CHROMA_COLLECTION, count)
    return _vectorstore


# ── Text splitter ─────────────────────────────────────────────────────────────
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a text splitter configured for research documents.
    Uses RecursiveCharacterTextSplitter which respects paragraph/sentence boundaries.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        separators      = ["\n\n", "\n", ". ", " ", ""],
    )


# ── Document ingestion ────────────────────────────────────────────────────────
def ingest_text(
    text    : str,
    source  : str = "manual",
    metadata: Optional[dict] = None,
) -> int:
    """
    Chunks and embeds raw text into the ChromaDB vector store.

    Args:
        text     : Raw text content to ingest.
        source   : Label for where this text came from (e.g. filename, URL).
        metadata : Optional extra metadata dict stored alongside each chunk.

    Returns:
        Number of chunks added to the vector store.
    """
    if not text or not text.strip():
        logger.warning("ingest_text called with empty text — skipping.")
        return 0

    splitter = get_text_splitter()
    base_meta = {"source": source, **(metadata or {})}

    # Build LangChain Document objects
    doc      = Document(page_content=text, metadata=base_meta)
    chunks   = splitter.split_documents([doc])

    # Add a unique ID to each chunk to prevent duplicate ingestion
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{source}_{i}_{chunk.page_content[:50]}".encode()).hexdigest()
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = i

    vs = get_vectorstore()
    vs.add_documents(chunks)

    logger.info("Ingested %d chunks from source: '%s'", len(chunks), source)
    return len(chunks)


def ingest_file(file_path: str) -> int:
    """
    Reads a .txt or .md file from disk and ingests it into ChromaDB.

    Args:
        file_path : Absolute or relative path to the text file.

    Returns:
        Number of chunks added, or 0 if file cannot be read.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        return 0

    if path.suffix.lower() not in {".txt", ".md"}:
        logger.warning("Unsupported file type: %s — only .txt and .md supported.", path.suffix)
        return 0

    try:
        text = path.read_text(encoding="utf-8")
        return ingest_text(text, source=path.name, metadata={"file_path": str(path)})
    except Exception as exc:
        logger.error("Failed to read file '%s': %s", file_path, exc)
        return 0


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = RAG_TOP_K) -> list[dict]:
    """
    Retrieves the top-k most relevant document chunks for a query.

    Args:
        query : The search query to match against stored documents.
        k     : Number of chunks to return.

    Returns:
        List of dicts: [{"content": ..., "source": ..., "score": ..., "chunk_index": ...}]
    """
    vs      = get_vectorstore()
    count   = vs._collection.count()

    if count == 0:
        logger.info("Vector store is empty — no documents to retrieve.")
        return []

    # similarity_search_with_relevance_scores returns (Document, score) tuples
    results = vs.similarity_search_with_relevance_scores(query, k=min(k, count))

    output = []
    for doc, score in results:
        output.append({
            "content"    : doc.page_content,
            "source"     : doc.metadata.get("source", "unknown"),
            "score"      : round(score, 3),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        })

    logger.info("Retrieved %d chunks for query: '%s'", len(output), query)
    return output


def get_collection_stats() -> dict:
    """
    Returns basic stats about what's currently stored in ChromaDB.

    Returns:
        {"total_chunks": int, "collection": str, "db_path": str}
    """
    vs    = get_vectorstore()
    count = vs._collection.count()
    return {
        "total_chunks": count,
        "collection"  : CHROMA_COLLECTION,
        "db_path"     : CHROMA_DB_PATH,
    }


def clear_collection() -> None:
    """
    Deletes all documents from the ChromaDB collection.
    Used for resetting between research sessions from the UI.
    """
    global _vectorstore
    vs = get_vectorstore()
    vs.delete_collection()
    _vectorstore = None
    logger.info("ChromaDB collection '%s' cleared.", CHROMA_COLLECTION)


# ── LangChain Tools (used by agents) ─────────────────────────────────────────

@tool
def retrieve_documents(query: str) -> str:
    """
    Search the local document knowledge base for information relevant to a query.
    Use this when the user has uploaded documents or when you need to find
    previously stored research content. Returns the most relevant passages.

    Args:
        query : The topic or question to search for in stored documents.

    Returns:
        Formatted text of the most relevant document passages with their sources.
    """
    logger.info("retrieve_documents tool called | query: '%s'", query)
    results = retrieve(query)

    if not results:
        return "No relevant documents found in the knowledge base. The document store may be empty — try using web_search instead."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] Source: {r['source']} (relevance: {r['score']})")
        lines.append(f"    {r['content'][:500]}")
        lines.append("")

    return "\n".join(lines).strip()


@tool
def add_to_knowledge_base(text: str, source: str = "agent_research") -> str:
    """
    Save a piece of text into the local knowledge base for future retrieval.
    Use this to store important findings, summaries, or research notes
    so they can be retrieved in later steps of the research workflow.

    Args:
        text   : The text content to store.
        source : A short label describing where this content came from.

    Returns:
        Confirmation message with the number of chunks stored.
    """
    logger.info("add_to_knowledge_base tool called | source: '%s'", source)
    count = ingest_text(text, source=source)

    if count == 0:
        return "Failed to add content to knowledge base — text was empty."

    return f"Successfully stored {count} chunk(s) from '{source}' into the knowledge base."


# ── Quick self-test (run: python src/tools/retrieval_tool.py) ─────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("\n" + "=" * 55)
    print("  Retrieval Tool Test")
    print("=" * 55)

    # 1. Check stats before
    stats = get_collection_stats()
    print(f"\n[1] Collection stats (before): {stats}")

    # 2. Ingest sample text
    sample = """
    LangGraph is a library for building stateful, multi-actor applications with LLMs.
    It extends LangChain by allowing you to define workflows as graphs where nodes
    are agents or functions and edges are transitions between them.
    LangGraph supports cycles, making it ideal for agentic loops where an agent
    thinks, acts, observes, and repeats until it reaches a goal.

    Key features of LangGraph include:
    - Stateful graph execution with persistent memory
    - Support for multi-agent orchestration
    - Human-in-the-loop checkpointing
    - Streaming of intermediate steps
    - Built-in support for tool calling
    """

    print("\n[2] Ingesting sample text...")
    count = ingest_text(sample, source="langgraph_overview")
    print(f"    Chunks ingested: {count}")

    # 3. Check stats after
    stats = get_collection_stats()
    print(f"\n[3] Collection stats (after):  {stats}")

    # 4. Retrieve
    query = "What is LangGraph used for?"
    print(f"\n[4] Retrieving for: '{query}'")
    results = retrieve(query, k=2)
    for r in results:
        print(f"  • [score: {r['score']}] {r['content'][:120]}...")

    # 5. LangChain tool test
    print(f"\n[5] retrieve_documents tool:")
    result = retrieve_documents.invoke(query)
    print(result[:400])

    # 6. Clear collection (clean state for next run)
    print("\n[6] Clearing collection...")
    clear_collection()
    stats = get_collection_stats()
    print(f"    Collection stats (cleared): {stats}")

    print("\n" + "=" * 55)
    print("Retrieval tool test complete.")
    print("=" * 55 + "\n")