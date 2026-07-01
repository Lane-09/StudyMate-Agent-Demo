from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMORY_DIR = PROJECT_ROOT / "data" / "processed" / "memory"
MEMORY_PATH = MEMORY_DIR / "user_memory.json"

TRACE_DIR = PROJECT_ROOT / "evaluation"
TRACE_PATH = TRACE_DIR / "agent_trace.log"


DEFAULT_MEMORY = {
    "language_preference": "English",
    "explanation_level": "general",
    "preferred_style": "clear and concise",
    "recent_topics": [],
    "total_queries": 0
}


def load_memory() -> Dict[str, Any]:
    """
    Load user memory from a local JSON file.
    If the file does not exist, create a default memory file.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if not MEMORY_PATH.exists():
        save_memory(DEFAULT_MEMORY)
        return DEFAULT_MEMORY.copy()

    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_MEMORY.copy()


def save_memory(memory: Dict[str, Any]) -> None:
    """
    Save user memory to a local JSON file.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def detect_topics(query: str) -> List[str]:
    """
    Detect learning topics from the user query.
    This is rule-based, simple, and transparent.
    """
    lower_query = query.lower()

    topic_keywords = {
        "Course Overview": [
            "overview", "course", "assessment", "dimensionality"
        ],
        "Transformer": [
            "transformer", "attention", "encoder", "decoder", "positional"
        ],
        "Vision Transformer": [
            "vit", "vision transformer", "patchify", "patch"
        ],
        "Self-supervised Learning": [
            "self-supervised", "ssl", "dino", "mae", "teacher", "student"
        ],
        "Multimodal Alignment": [
            "alignment", "shared embedding", "zero-shot", "retrieval"
        ],
        "CLIP": [
            "clip", "contrastive language-image"
        ],
        "LLaVA": [
            "llava", "visual instruction", "connector"
        ],
        "MLLM": [
            "mllm", "multimodal large language model", "vision-language"
        ],
        "Indexing": [
            "indexing", "vector database", "chroma", "ann", "knn", "hnsw"
        ],
        "Token Reduction": [
            "token reduction", "information bottleneck", "pruning", "merging"
        ],
        "Multimodal RAG": [
            "rag", "retrieval augmented generation", "chunking"
        ],
        "RAG Evaluation": [
            "evaluation", "hallucination", "ragas", "faithfulness", "groundedness"
        ],
        "Agents": [
            "agent", "react", "tool", "memory", "planning"
        ],
        "PEFT": [
            "peft", "parameter-efficient", "parameter efficient",
            "fine-tuning", "fine tuning"
        ],
        "SFT": [
            "sft", "supervised fine-tuning", "supervised finetuning"
        ],
        "Prompt Tuning": [
            "prompt tuning", "soft prompt", "learnable prompt", "p-tuning"
        ],
        "Adapter": [
            "adapter", "adapterfusion", "houlsby"
        ],
        "LoRA": [
            "lora", "low-rank adaptation", "low rank adaptation"
        ],
        "QLoRA": [
            "qlora", "quantized", "4-bit", "4 bit"
        ],
        "DoRA": [
            "dora", "weight-decomposed", "weight decomposed"
        ],
        "OCR": [
            "ocr", "image text", "screenshot", "diagram"
        ]
    }

    detected = []

    for topic, keywords in topic_keywords.items():
        if any(keyword in lower_query for keyword in keywords):
            detected.append(topic)

    return detected


def update_memory_from_query(query: str) -> Dict[str, Any]:
    """
    Update memory based on the current query.
    """
    memory = load_memory()
    lower_query = query.lower()

    if "中文" in query or "chinese" in lower_query or "用中文" in query:
        memory["language_preference"] = "Chinese"

    if (
        "小白" in query
        or "beginner" in lower_query
        or "step by step" in lower_query
        or "一步一步" in query
    ):
        memory["explanation_level"] = "beginner"
        memory["preferred_style"] = "step-by-step and beginner-friendly"

    detected_topics = detect_topics(query)

    recent_topics = memory.get("recent_topics", [])

    for topic in detected_topics:
        if topic not in recent_topics:
            recent_topics.append(topic)

    memory["recent_topics"] = recent_topics[-10:]
    memory["total_queries"] = int(memory.get("total_queries", 0)) + 1
    memory["last_updated"] = datetime.now().isoformat(timespec="seconds")

    save_memory(memory)
    return memory


def build_style_instruction_from_memory(memory: Dict[str, Any]) -> str:
    """
    Convert stored memory into a style instruction for the LLM.
    """
    instructions = []

    if memory.get("language_preference") == "Chinese":
        instructions.append("Answer in Chinese when appropriate.")

    if memory.get("explanation_level") == "beginner":
        instructions.append("Explain in a beginner-friendly and step-by-step way.")

    preferred_style = memory.get("preferred_style", "")
    if preferred_style:
        instructions.append(f"Preferred style: {preferred_style}.")

    recent_topics = memory.get("recent_topics", [])
    if recent_topics:
        instructions.append(
            "The user's recent learning topics include: "
            + ", ".join(recent_topics)
            + "."
        )

    return " ".join(instructions)


def append_agent_trace(trace_record: Dict[str, Any]) -> None:
    """
    Append one agent run record to evaluation/agent_trace.log.
    Each line is a JSON object.
    """
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    trace_record["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")