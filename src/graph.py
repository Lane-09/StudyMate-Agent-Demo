from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from retriever import query_knowledge_base, format_sources
from llm import generate_answer
from memory_store import (
    load_memory,
    update_memory_from_query,
    build_style_instruction_from_memory,
    append_agent_trace
)


class StudyMateState(TypedDict):
    query: str
    route: str
    style_instruction: str
    memory: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    tool_calls: List[str]
    trace_steps: List[str]


def load_user_memory(state: StudyMateState) -> Dict[str, Any]:
    """
    Load existing user memory before routing.
    """
    memory = load_memory()

    return {
        "memory": memory,
        "trace_steps": state.get("trace_steps", []) + ["load_user_memory"]
    }


def route_query(state: StudyMateState) -> Dict[str, Any]:
    """
    Route the query based on simple transparent rules.
    """
    query = state["query"]
    lower_query = query.lower()
    memory = state.get("memory", {})

    route = "general_lecture_query"

    visual_keywords = [
        "pdf", "page", "figure", "diagram", "image", "screenshot",
        "chart", "table", "ocr", "slide", "structure", "architecture",
        "transformer structure", "attention diagram",
        "图片", "截图", "图", "图表", "表格", "结构图", "架构图", "第几页", "页面"
    ]

    analytical_keywords = [
        "compare", "difference", "why", "how", "relationship",
        "evaluate", "limitation", "advantage", "disadvantage",
        "对比", "区别", "为什么", "如何", "关系", "评价", "局限"
    ]

    agent_keywords = [
        "agent", "tool", "react", "planning", "memory",
        "智能体", "工具", "规划", "记忆"
    ]

    peft_keywords = [
        "peft", "sft", "fine-tuning", "fine tuning",
        "prompt tuning", "adapter", "adapterfusion",
        "lora", "qlora", "dora",
        "parameter-efficient", "parameter efficient",
        "参数高效", "微调"
    ]

    if any(keyword in lower_query for keyword in visual_keywords):
        route = "visual_or_pdf_ocr_query"

    if any(keyword in lower_query for keyword in analytical_keywords):
        route = "analytical_query"

    if any(keyword in lower_query for keyword in agent_keywords):
        route = "agent_related_query"

    if any(keyword in lower_query for keyword in peft_keywords):
        route = "peft_related_query"

    style_instruction = build_style_instruction_from_memory(memory)

    beginner_keywords = [
        "beginner", "step by step", "simple", "explain clearly",
        "小白", "详细", "一步一步", "简单解释"
    ]

    chinese_keywords = [
        "中文", "用中文", "Chinese"
    ]

    if any(keyword.lower() in lower_query for keyword in beginner_keywords):
        style_instruction += " Explain in a beginner-friendly way with clear steps."

    if any(keyword.lower() in lower_query for keyword in chinese_keywords):
        style_instruction += " Answer in Chinese."

    return {
        "route": route,
        "style_instruction": style_instruction.strip(),
        "trace_steps": state.get("trace_steps", []) + [f"route_query:{route}"]
    }


def retrieve_context(state: StudyMateState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from ChromaDB.
    """
    query = state["query"]
    route = state["route"]

    n_results = 5

    if route == "analytical_query":
        n_results = 7

    if route == "agent_related_query":
        n_results = 7

    if route == "peft_related_query":
        n_results = 7

    retrieved_docs = query_knowledge_base(
        query=query,
        n_results=n_results,
        allowed_modalities=None
    )

    sources = format_sources(retrieved_docs)

    return {
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "tool_calls": state.get("tool_calls", []) + ["query_knowledge_base"],
        "trace_steps": state.get("trace_steps", []) + [f"retrieve_context:n_results={n_results}"]
    }


def generate_final_answer(state: StudyMateState) -> Dict[str, Any]:
    """
    Generate final answer using retrieved context.
    """
    answer = generate_answer(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        style_instruction=state["style_instruction"]
    )

    return {
        "answer": answer,
        "tool_calls": state.get("tool_calls", []) + ["ollama_generate_answer"],
        "trace_steps": state.get("trace_steps", []) + ["generate_final_answer"]
    }


def update_memory_and_trace(state: StudyMateState) -> Dict[str, Any]:
    """
    Update user memory and write agent trace log.
    """
    updated_memory = update_memory_from_query(state["query"])

    trace_record = {
        "query": state["query"],
        "route": state.get("route", ""),
        "sources": state.get("sources", []),
        "tool_calls": state.get("tool_calls", []),
        "trace_steps": state.get("trace_steps", []),
        "memory_after_query": updated_memory
    }

    append_agent_trace(trace_record)

    return {
        "memory": updated_memory,
        "trace_steps": state.get("trace_steps", []) + ["update_memory_and_trace"]
    }


def build_graph():
    """
    Build LangGraph workflow.
    """
    workflow = StateGraph(StudyMateState)

    workflow.add_node("load_user_memory", load_user_memory)
    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_final_answer", generate_final_answer)
    workflow.add_node("update_memory_and_trace", update_memory_and_trace)

    workflow.set_entry_point("load_user_memory")

    workflow.add_edge("load_user_memory", "route_query")
    workflow.add_edge("route_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_final_answer")
    workflow.add_edge("generate_final_answer", "update_memory_and_trace")
    workflow.add_edge("update_memory_and_trace", END)

    return workflow.compile()


def run_agent(query: str) -> Dict[str, Any]:
    """
    Run StudyMate Agent.
    """
    app = build_graph()

    initial_state: StudyMateState = {
        "query": query,
        "route": "",
        "style_instruction": "",
        "memory": {},
        "retrieved_docs": [],
        "answer": "",
        "sources": [],
        "tool_calls": [],
        "trace_steps": []
    }

    result = app.invoke(initial_state)

    return result


def main() -> None:
    """
    Terminal test.
    """
    query = "我是小白，请用中文一步一步解释 Lecture 11 中 PEFT 和 LoRA 的关系。"

    result = run_agent(query)

    print("\nQuestion:")
    print(query)

    print("\nRoute:")
    print(result.get("route"))

    print("\nAnswer:")
    print(result.get("answer"))

    print("\nSources:")
    for source in result.get("sources", []):
        print("-", source)

    print("\nTool calls:")
    for tool in result.get("tool_calls", []):
        print("-", tool)

    print("\nTrace steps:")
    for step in result.get("trace_steps", []):
        print("-", step)


if __name__ == "__main__":
    main()