import re
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama


MODEL_NAME = "qwen3:0.6b-q4_K_M"


def clean_model_output(text: str) -> str:
    """
    Some local models may output hidden thinking tags.
    This function removes <think>...</think> content if it appears.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def get_llm() -> ChatOllama:
    """
    Create an Ollama chat model.
    Make sure Ollama is installed and the model has been pulled.
    """
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0.2
    )


def build_context_text(retrieved_docs: Optional[List[Dict[str, Any]]]) -> str:
    """
    Convert retrieved documents into a single context string.
    """
    if not retrieved_docs:
        return "No retrieved context."

    context_parts = []

    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})

        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "")
        modality = metadata.get("modality", "unknown")
        text = doc.get("text", "")

        header = f"[Source: {source_file}"
        if page_number and page_number != "unknown":
            header += f", Page: {page_number}"
        header += f", Modality: {modality}]"

        context_parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    query: str,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    style_instruction: str = ""
) -> str:
    """
    Generate an answer using retrieved context.
    """
    context_text = build_context_text(retrieved_docs)

    prompt = f"""
You are StudyMate Agent, a personalised multimodal study assistant.

Your task:
Answer the user's question using the retrieved context.

Rules:
1. Use the retrieved context as the main evidence.
2. If the context is not enough, say: "The retrieved evidence is insufficient."
3. Mention the source file and page number when available.
4. Explain clearly and simply.
5. Do not invent unsupported details.

Style instruction:
{style_instruction}

Retrieved context:
{context_text}

User question:
{query}

Final answer:
""".strip()

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return clean_model_output(answer)

    except Exception as e:
        return (
            "LLM generation failed.\n\n"
            f"Error: {e}\n\n"
            "Please check whether Ollama is installed, running, "
            f"and whether the model '{MODEL_NAME}' has been pulled."
        )


def generate_plain_answer(query: str) -> str:
    """
    Plain LLM baseline.
    This version does not use retrieval context.
    """
    prompt = f"""
You are a helpful study assistant.

Answer the following question using your general knowledge only.
Do not use any personal knowledge base.

User question:
{query}

Final answer:
""".strip()

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return clean_model_output(answer)

    except Exception as e:
        return (
            "Plain LLM generation failed.\n\n"
            f"Error: {e}\n\n"
            "Please check Ollama and the local model."
        )


def main() -> None:
    """
    Simple terminal test.
    """
    query = "What is a Transformer?"
    answer = generate_plain_answer(query)

    print("\nQuestion:")
    print(query)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()