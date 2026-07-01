from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "chroma"
COLLECTION_NAME = "studymate_knowledge_base"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def get_collection():
    """
    Open the existing ChromaDB collection.
    This function does not rebuild the database.
    It only connects to the index created by build_index.py.
    """
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector database not found: {VECTOR_DB_DIR}\n"
            "Please run: python src\\build_index.py"
        )

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection


def query_knowledge_base(
    query: str,
    n_results: int = 5,
    allowed_modalities: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search the knowledge base.

    Args:
        query: User question.
        n_results: Number of final results to return.
        allowed_modalities:
            None means search all records.
            Example: ["text"] means only keep raw txt/docx text records.

    Returns:
        A list of retrieved records.
    """
    collection = get_collection()

    total_count = collection.count()

    if total_count == 0:
        return []

    # If we need modality filtering, retrieve more first and filter later.
    if allowed_modalities:
        raw_n_results = min(max(n_results * 5, 20), total_count)
    else:
        raw_n_results = min(n_results, total_count)

    results = collection.query(
        query_texts=[query],
        n_results=raw_n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for i, document in enumerate(documents):
        metadata = metadatas[i]
        distance = distances[i] if i < len(distances) else None

        modality = metadata.get("modality", "")

        if allowed_modalities and modality not in allowed_modalities:
            continue

        item = {
            "rank": len(retrieved) + 1,
            "text": document,
            "metadata": metadata,
            "distance": distance
        }

        retrieved.append(item)

        if len(retrieved) >= n_results:
            break

    return retrieved


def format_sources(retrieved_docs: List[Dict[str, Any]]) -> List[str]:
    """
    Convert retrieved documents into readable source strings.
    """
    sources = []

    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})

        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "")
        modality = metadata.get("modality", "unknown")

        if page_number and page_number != "unknown":
            source = f"{source_file}, page {page_number}, modality: {modality}"
        else:
            source = f"{source_file}, modality: {modality}"

        if source not in sources:
            sources.append(source)

    return sources


def print_retrieval_results(query: str, retrieved_docs: List[Dict[str, Any]]) -> None:
    """
    Print search results in terminal for testing.
    """
    print("\nUser query:")
    print(query)

    if not retrieved_docs:
        print("\nNo relevant documents found.")
        return

    print("\nRetrieved results:")

    for doc in retrieved_docs:
        metadata = doc["metadata"]

        print("\n--------------------------------")
        print(f"Rank: {doc['rank']}")
        print(f"Source file: {metadata.get('source_file')}")
        print(f"Page number: {metadata.get('page_number')}")
        print(f"Modality: {metadata.get('modality')}")
        print(f"Distance: {doc.get('distance')}")
        print("Text preview:")
        print(doc["text"][:600])


def main() -> None:
    """
    Simple terminal test.
    """
    query = "What components are shown in the Transformer structure image?"

    retrieved_docs = query_knowledge_base(
        query=query,
        n_results=5
    )

    print_retrieval_results(query, retrieved_docs)


if __name__ == "__main__":
    main()