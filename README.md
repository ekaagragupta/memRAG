# MemoRAG

**MemoRAG** is a memory-augmented retrieval-augmented generation (RAG) framework that overcomes the core limitations of standard RAG systems when handling long, complex documents. By first building a compressed global memory of the full context, MemoRAG enables the retriever to be *globally aware* — generating precise clues and surrogate queries before retrieval, resulting in dramatically more accurate and complete answers.

---

## Why MemoRAG?

Standard RAG systems retrieve document chunks directly from a query. This works for simple lookups but fails for:

- **Questions requiring global context** — e.g. "Which year had peak revenue?" when peaks span multiple sections
- **High-level aggregation** — e.g. "How did the pandemic impact the company?" requires synthesizing many passages
- **Open-ended summarization** — summary instructions aren't directly searchable

MemoRAG addresses all three failure modes by using a lightweight memory model to first *recall* relevant clues and draft candidate answers, then retrieve precise evidence, then generate the final response.

---

## Architecture

### Standard RAG vs MemoRAG

| Step | Standard RAG | MemoRAG |
|------|-------------|---------|
| Input | Query → Retriever → Evidence → Answer | Query → Memory → Clues + Draft → Retriever → Evidence → Answer |
| Retrieval quality | Query-local | Globally-aware |
| Summarization | Fails (no direct anchor) | Works (memory generates key points first) |
| Complex QA | Often wrong/incomplete | Accurate |

### Core Pipeline

```
Long Context
    │
    ▼
[Memory Model]  ──── Stores compressed KV-cache of entire context
    │
    ├──► recall(query)    → text spans from memory
    ├──► rewrite(query)   → surrogate questions
    └──► answer(query)    → draft answer (optional)
         │
         ▼
[Dense Retriever (FAISS + BGE-M3)]
    │   Uses clues + surrogate queries + draft answer as retrieval queries
    ▼
[Retrieved Evidence Chunks]
    │
    ▼
[Generator LLM]  ──── Produces final answer grounded in evidence
```

---

## Variants

### `MemoRAG` (Full)

Uses a dedicated memory model (LLM with long context window, e.g. Llama 3.1 8B or a custom `memorag` beacon model) to encode the full context into KV-cache. Best for maximum quality on very long documents.

```python
from memorag import MemoRAG

model = MemoRAG(
    mem_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    ret_model_name_or_path="BAAI/bge-m3",
    cache_dir="path_to_model_cache",       # optional
    access_token="hugging_face_token"      # optional
)

context = open("document.txt").read()
model.memorize(context, save_dir="my_cache", print_stats=True)

# QA using memory only
answer = model(context, "What are the main themes?", task_type="qa")

# Full MemoRAG pipeline (memory → retrieval → generation)
answer = model(context, "What are the main themes?", task_type="memorag")

# Summarization
summary = model(context, task_type="summarize")
```

### `MemoRAGLite`

A lightweight variant using `Qwen2.5-1.5B-Instruct` as the memory/generation model. Significantly lower GPU memory requirements while retaining the core MemoRAG pipeline. Automatically adapts batch size based on available GPU memory.

```python
from memorag import MemoRAGLite

pipe = MemoRAGLite(
    gen_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",  # default
    ret_model_name_or_path="BAAI/bge-m3",
    ret_hit=3,
    retrieval_chunk_size=512,
    load_in_4bit=False,
    enable_flash_attn=True
)

context = open("document.txt").read()
pipe.memorize(context, save_dir="my_cache", print_stats=True)

# Load pre-cached memory
pipe.load("my_cache")

# Query
print(pipe("What is the book's main theme?"))
```

### `Agent` (API-backed Generator)

Allows using OpenAI, Azure OpenAI, or DeepSeek as the generator model instead of a local LLM. Pass the agent as `customized_gen_model` to `MemoRAG` or `MemoRAGLite`.

```python
from memorag import Agent

agent = Agent(
    model="gpt-4o",
    source="openai",       # "openai" | "azure" | "deepseek"
    api_dict={"api_key": "sk-..."},
    temperature=0.0
)

# Use with MemoRAGLite
from memorag import MemoRAGLite
pipe = MemoRAGLite(customized_gen_model=agent)
```

---

## Memory Formation

The `memorize()` method processes the full context once and caches it for repeated queries:

1. **Gist extraction** — the context is split into chunks; the model generates compressed summaries ("gists") of each
2. **KV-cache encoding** — the concatenated gists are encoded into a `DynamicCache` (transformers KV-cache), stored in memory
3. **Dense retrieval index** — the full raw context is chunked and indexed with FAISS using BGE-M3 embeddings

The cached memory is saved as:

```
save_dir/
├── memory.bin     # PyTorch KV-cache + prompts + language metadata
├── index.bin      # FAISS dense retrieval index
└── chunks.json    # Raw text chunks for retrieval
```

Memory file sizes:
- `MemoRAG` (Llama 3.1 8B, ~128K tokens): ~11–15 GB
- `MemoRAGLite` (Qwen2.5-1.5B, ~122K tokens): ~0.29 GB

---

## Retrieval

`DenseRetriever` wraps a BGE-M3 encoder with a FAISS flat index (cosine similarity). During the MemoRAG pipeline, retrieval queries are not just the raw user query — they include:

- Text spans recalled from memory
- Surrogate questions rewritten from memory
- Optionally: a draft answer generated from memory

This multi-query retrieval dramatically improves recall for complex, globally-scoped questions.

---

## Prompts

Prompt templates are defined in `prompt.py`. Available prompt keys:

| Key | Purpose |
|-----|---------|
| `context` | System prompt for context ingestion |
| `gist` | Compression/summarization of context chunks |
| `span` | Recall specific text spans from memory |
| `sur` | Generate surrogate/clue questions |
| `qa` | Direct QA from memory |
| `sum` | Generate key point list for summarization |
| `qa_gen` | Final answer generation given retrieved evidence |
| `sum_gen` | Final summary generation given retrieved evidence |

---

## Installation

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
transformers>=4.43.1
deepspeed>=0.14.0
minference==0.1.5
triton==2.3.1
accelerate
datasets
rouge
fuzzywuzzy
jieba
python-Levenshtein
seaborn
tiktoken
openai
semantic_text_splitter
langdetect
```

**Hardware requirements:**
- `MemoRAGLite`: minimum 24 GiB GPU VRAM recommended (adaptive batch sizing for lower memory)
- `MemoRAG` (full): 40+ GiB GPU VRAM recommended for 8B models without 4-bit quantization
- 4-bit quantization (`load_in_4bit=True`) significantly reduces VRAM usage

---

## Supported Models

| Role | Recommended Models |
|------|-------------------|
| Memory (Full) | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Memory (Lite) | `Qwen/Qwen2.5-1.5B-Instruct` |
| Retrieval | `BAAI/bge-m3` |
| Generator (API) | OpenAI GPT-4o, Azure OpenAI, DeepSeek |

Custom memory models with the `memorag` identifier in their path are treated as beacon models; all others use the long-context KV-cache approach.

---

## Demo

A web demo (shown below) is available with support for uploading documents (TXT, PDF, DOCX up to 200MB), selecting pre-cached corpora, and configuring generation parameters.

Key demo settings:
- Max Generation Tokens (10–1024)
- Temperature (0.0–2.0)
- Generation Model selection (e.g. Mistral-7B-Instruct-v0.2)
- File upload or cached corpus selection (e.g. Harry Potter)

---

## Module Reference

```
memorag/
├── __init__.py          # Exports: Memory, MemoRAG, MemoRAGLite, Model, Agent
├── memorag.py           # Model, Memory, MemoRAG classes
├── memorag_lite.py      # MemoRAGLite class
├── agent.py             # Agent class (OpenAI / Azure / DeepSeek)
├── retrieval.py         # DenseRetriever, FaissIndex
└── prompt.py            # en_prompts
```

### Key classes

**`Model`** — Base LLM wrapper. Handles tokenization, chat template formatting, batched generation. Supports flash attention, 4-bit quantization, and MInference patching for long-context efficiency.

**`Memory(Model)`** — Extends `Model` with `memorize()`, `generate_w_memory()`, `recall()`, `rewrite()`, `answer()`, `summarize()`, `save()`, `load()`. Supports both beacon-style and long-context KV-cache memory types.

**`MemoRAG`** — Full pipeline. Composes `Memory` + `DenseRetriever` + optional separate `Model` for generation. Handles `qa`, `memorag`, and `summarize` task types.

**`MemoRAGLite`** — Lightweight pipeline. Uses a small generative model for both memory formation and generation. Adapts batch size to available GPU memory, supports caching and loading.

**`DenseRetriever`** — Dense bi-encoder retriever using any HuggingFace encoder (default: BGE-M3). Builds and queries a FAISS flat index with cosine similarity.

**`Agent`** — Thin wrapper around the OpenAI Python SDK supporting OpenAI, Azure OpenAI, and DeepSeek endpoints. Implements the same `.generate()` interface as `Model` for drop-in use as a generator.

---

## Example: English Book QA

```python
from memorag import MemoRAGLite

pipe = MemoRAGLite()
context = open("harry_potter.txt").read()
pipe.memorize(context, save_dir="harry_potter", print_stats=True)
# Detected language: en
# Context length: 122591 tokens
# Memory file size: 0.29 GB
# Number of chunks in retrieval corpus: 268

print(pipe("How are the mutual relationships of the main roles?"))
```

**Output:**
```
Harry Potter's closest friends are Ron Weasley and Hermione Granger. Together,
they form an inseparable trio throughout their years at Hogwarts, facing
countless dangers and challenges. Their friendship is built on mutual trust,
loyalty, and support, with Ron providing steadfast companionship and humor, and
Hermione contributing her intelligence and resourcefulness...
```



