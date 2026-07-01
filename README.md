# StudyMate-Agent-Demo
This is a public demo version. Original course PDFs are excluded for copyright and privacy reasons. The repository includes source code, evaluation results, screenshots, and sample data for demonstration.

---

# StudyMate Agent

StudyMate Agent is a personalised multimodal study assistant for INFS4205/7205 lecture learning.

The project uses lecture PDFs, PDF page-level OCR, image OCR, ChromaDB retrieval, LangGraph agent workflow, local Ollama answer generation, memory tracking, and quantitative evaluation.

---

## Project Overview

StudyMate Agent is designed to help students ask questions about their own lecture materials.

A plain LLM can answer general questions, but it cannot reliably use a student's own course materials as evidence. This project therefore builds a personalised PDF + OCR knowledge base and compares three systems:

1. Plain LLM
2. PDF + OCR RAG
3. Final PDF + OCR Agent

The final system retrieves relevant lecture evidence and displays source pages for detailed reference.

---

## Knowledge Base

The personalised knowledge base contains INFS4205/7205 lecture materials from Lecture 1 to Lecture 11, one Transformer structure image, and a personal learning profile.

Included files:

- `Lecture1_Overview.pdf`
- `Lecture2_Multimodal_Embedding.pdf`
- `Lecture3_Self_Supervised_Multimodal_Embedding.pdf`
- `Lecture4_Multimodal_Alignment.pdf`
- `Lecture5_Multimodal_LLM.pdf`
- `Lecture6_High_Dimensional_Indexing.pdf`
- `Lecture7_Token_Reduction.pdf`
- `Lecture8_Multimodal_RAG.pdf`
- `Lecture9_Multimodal_RAG_Evaluation.pdf`
- `Lecture10_Agents.pdf`
- `Lecture11_PEFT.pdf`
- `Transformer_Structure.png`
- `my_learning_profile.txt`

The system treats PDFs as multimodal documents. Each PDF page is processed using embedded text extraction and page-level OCR. Therefore, the system supports both document text and visual-text information from figures, diagrams, tables, screenshots, and slide layouts.

---

## Main Topics

The knowledge base covers:

- High-dimensional data
- Transformer architecture
- Tokenization
- Positional encoding
- Self-attention
- Multi-head attention
- Vision Transformer
- Patchify operation
- Multimodal embedding
- Multimodal alignment
- CLIP
- CoCa
- BLIP-2
- LLaVA
- Multimodal large language models
- High-dimensional indexing
- Vector database
- Token reduction
- Multimodal RAG
- RAG evaluation
- Agents
- ReAct
- Memory and tool use
- PEFT
- SFT
- Prompt tuning
- Adapter-based PEFT
- LoRA
- QLoRA
- DoRA

---

## Source Code Repository and Reproducibility

This repository contains the source code, installation instructions, dependencies, run instructions, evaluation files, and report files for the StudyMate Agent project.

### Source Code

The main source code is stored in the `src/` folder.

| File | Purpose |
|---|---|
| `src/pdf_processor.py` | Extracts PDF embedded text and performs page-level OCR |
| `src/build_index.py` | Builds the ChromaDB vector database from processed knowledge base files |
| `src/retriever.py` | Retrieves relevant chunks from ChromaDB |
| `src/llm.py` | Connects to Ollama and generates answers |
| `src/graph.py` | Defines the LangGraph agent workflow |
| `src/memory_store.py` | Stores user memory and writes agent trace logs |
| `src/app.py` | Provides the Streamlit web interface |
| `src/evaluator.py` | Runs quantitative evaluation and generates `results.csv` |

---

## Project Structure

```text
A3_StudyMate_Agent/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── documents/
│   │   │   ├── Lecture1_Overview.pdf
│   │   │   ├── Lecture2_Multimodal_Embedding.pdf
│   │   │   ├── Lecture3_Self_Supervised_Multimodal_Embedding.pdf
│   │   │   ├── Lecture4_Multimodal_Alignment.pdf
│   │   │   ├── Lecture5_Multimodal_LLM.pdf
│   │   │   ├── Lecture6_High_Dimensional_Indexing.pdf
│   │   │   ├── Lecture7_Token_Reduction.pdf
│   │   │   ├── Lecture8_Multimodal_RAG.pdf
│   │   │   ├── Lecture9_Multimodal_RAG_Evaluation.pdf
│   │   │   ├── Lecture10_Agents.pdf
│   │   │   └── Lecture11_PEFT.pdf
│   │   │
│   │   ├── images/
│   │   │   └── Transformer_Structure.png
│   │   │
│   │   └── text/
│   │       └── my_learning_profile.txt
│   │
│   └── processed/
│       ├── extracted_text/
│       ├── pdf_pages/
│       └── memory/
│
├── vector_db/
│   └── chroma/
│
├── src/
│   ├── pdf_processor.py
│   ├── build_index.py
│   ├── retriever.py
│   ├── llm.py
│   ├── graph.py
│   ├── memory_store.py
│   ├── app.py
│   └── evaluator.py
│
├── evaluation/
│   ├── test_queries.csv
│   ├── results.csv
│   └── agent_trace.log
│
└── report/
    └── report.md
```

---

## Dependencies

This project requires both Python packages and external software.

### Python Dependencies

Python dependencies are listed in `requirements.txt`.

Main Python dependencies include:

| Dependency | Purpose |
|---|---|
| `streamlit` | Web interface |
| `chromadb` | Vector database |
| `sentence-transformers` | Text embedding model |
| `pymupdf` | PDF text extraction and PDF page rendering |
| `pytesseract` | Python wrapper for Tesseract OCR |
| `pillow` | Image processing |
| `pandas` | Evaluation data processing |
| `langgraph` | Agent workflow |
| `langchain` | LLM framework support |
| `langchain-ollama` | Connects Python with Ollama |
| `python-docx` | Optional Word document support |

### External Software Dependencies

| Software | Purpose |
|---|---|
| Tesseract OCR | Recognises text from rendered PDF pages and images |
| Ollama | Runs the local LLM for answer generation |

### Local Models

| Model | Purpose |
|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model for ChromaDB retrieval |
| `qwen3:0.6b-q4_K_M` | Local LLM used through Ollama |

---

## Installation Instructions

### 1. Create a virtual environment

```cmd
python -m venv .venv
```

### 2. Activate the virtual environment

```cmd
.venv\Scripts\activate.bat
```

After activation, the terminal should show:

```text
(.venv)
```

### 3. Install Python dependencies

```cmd
python -m pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install Tesseract OCR on Windows.

Recommended installation path:

```text
C:\Program Files\Tesseract-OCR
```

Check whether Tesseract is available:

```cmd
tesseract --version
```

If the command does not work, add the Tesseract installation folder to the system PATH.

### 5. Install Ollama

Install Ollama and pull the required local model:

```cmd
ollama pull qwen3:0.6b-q4_K_M
```

Check whether the model is installed:

```cmd
ollama list
```

The retrieval components can run without Ollama, but answer generation through LangGraph requires Ollama.

---

## Run Instructions

The project should be run in the following order.

### Step 1: Add knowledge base files

Put PDF files into:

```text
data/raw/documents/
```

Expected PDF files:

```text
Lecture1_Overview.pdf
Lecture2_Multimodal_Embedding.pdf
Lecture3_Self_Supervised_Multimodal_Embedding.pdf
Lecture4_Multimodal_Alignment.pdf
Lecture5_Multimodal_LLM.pdf
Lecture6_High_Dimensional_Indexing.pdf
Lecture7_Token_Reduction.pdf
Lecture8_Multimodal_RAG.pdf
Lecture9_Multimodal_RAG_Evaluation.pdf
Lecture10_Agents.pdf
Lecture11_PEFT.pdf
```

Put the image file into:

```text
data/raw/images/
```

Expected image:

```text
Transformer_Structure.png
```

Put the personal learning profile into:

```text
data/raw/text/
```

Expected text file:

```text
my_learning_profile.txt
```

### Step 2: Process PDF files and OCR pages

```cmd
python src\pdf_processor.py
```

This generates:

```text
data/processed/pdf_pages/
data/processed/extracted_text/
```

### Step 3: Build the vector database

```cmd
python src\build_index.py
```

This generates:

```text
vector_db/chroma/
```

### Step 4: Test the retriever

```cmd
python src\retriever.py
```

### Step 5: Test the LangGraph agent

```cmd
python src\graph.py
```

### Step 6: Run the Streamlit web application

```cmd
streamlit run src\app.py
```

The web app allows users to:

- ask questions
- view generated answers
- view retrieved PDF source pages
- view retrieved text evidence
- view agent trace
- view conversation history

### Step 7: Run quantitative evaluation

```cmd
python src\evaluator.py
```

The evaluation result will be saved to:

```text
evaluation/results.csv
```

The agent trace log will be saved to:

```text
evaluation/agent_trace.log
```

The user memory file will be saved to:

```text
data/processed/memory/user_memory.json
```

---

## Agent Memory and Trace

The final LangGraph agent includes a lightweight memory module.

It stores:

- language preference
- explanation level
- preferred answer style
- recent learning topics
- number of queries

The memory file is saved to:

```text
data/processed/memory/user_memory.json
```

The agent also records tool usage and routing traces in:

```text
evaluation/agent_trace.log
```

Each trace includes:

- user query
- selected route
- retrieved sources
- tool calls
- trace steps
- memory after query

---

## System Variants

The evaluation compares three systems.

| System | Description |
|---|---|
| Plain LLM | Uses Ollama directly without retrieval |
| PDF + OCR RAG | Uses ChromaDB retrieval over extracted PDF text, PDF OCR text, and image OCR text |
| Final PDF + OCR Agent | Uses LangGraph routing, memory, trace logging, ChromaDB retrieval, and Ollama answer generation |

---

## Evaluation

The evaluation uses test questions stored in:

```text
evaluation/test_queries.csv
```

The evaluation output is saved to:

```text
evaluation/results.csv
```

The project uses the following metrics:

| Metric | Meaning |
|---|---|
| Source Hit Rate | Whether the expected source file appears in the retrieved sources |
| Keyword Score | The proportion of expected keywords appearing in the answer |
| Answer Success Rate | Whether the answer reaches a keyword score of at least 0.5 |
| Average Latency | The average time required to generate an answer |

Plain LLM does not use retrieval, so Source Hit Rate is not directly applicable to this baseline.

---

## Example Questions

```text
What is a Transformer according to Lecture 2?
```

```text
What is multimodal alignment according to Lecture 4?
```

```text
What is LLaVA according to Lecture 5?
```

```text
Explain the Transformer structure image.
```

```text
What is LoRA according to Lecture 11?
```

```text
Explain PEFT and LoRA in beginner-friendly language.
```

---

## Limitations

This system uses OCR-based visual text extraction. It can recognise text in slides, diagrams, tables, and screenshots, but it does not use a true vision-language model to deeply understand images without text.

The system can display retrieved PDF source pages for detailed reference, but it may not fully understand visual relationships such as arrows, layout, and spatial structure inside complex diagrams.

The expanded Lecture 1-11 knowledge base improves topic coverage, but it may also increase retrieval difficulty because different lectures contain overlapping concepts.

Future work could add:

- stronger reranking
- better chunking
- multimodal page embeddings
- LLM-as-judge groundedness scoring
- a true vision-language model for image captioning and visual reasoning
