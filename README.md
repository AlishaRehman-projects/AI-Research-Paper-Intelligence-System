# 📚 AI Research Paper Intelligence System

An NLP pipeline for **semantic search**, **summarization**, and **Keyword extraction and scientific entity recognition** of ML research papers using sentence embeddings, FAISS, transformer-based summarization, and scientific NER.

Give it a query like *"Deep Learning in Medical Imaging"* and it retrieves the most relevant papers, summarizes each abstract, and extracts key scientific entities (methods, tasks, etc.) — all in one call.

## Features
- 🔎 Semantic search via `all-MiniLM-L6-v2` embeddings + FAISS cosine similarity
- 📝 Abstractive summarization with `facebook/bart-large-cnn`
- 🏷️ Keyword extraction with KeyBERT
- 🧬 Scientific NER (Method/Task tags) via `RJuro/SciNERTopic`
- ⚡ Cached embeddings/index so you only compute once
- 📦 Returns structured `list[dict]` results

## Dataset & Models
- Dataset: [`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (first 15,000 papers)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Summarization: `facebook/bart-large-cnn`
- Keywords: KeyBERT
- NER: `RJuro/SciNERTopic`
  
## Architecture
Query → Embedding (MiniLM) → FAISS Search → Top-K Papers → 
Summarization (BART) → Keyword Extraction (KeyBERT) → NER (SciNERTopic)
→ Structured Output

## Usage

```python
results = search_summarize_and_tag("Deep Learning in Medical Imaging", k=5)

for r in results:
    print(r["title"], r["similarity_score"])
    print(r["summary"])
    print(r["entities"])
```

Each result:
```python
{
    "title": "A Perspective on Deep Imaging",
    "similarity_score": 0.7355,
    "abstract": "...",
    "summary": "...",
    "entities": [{"text": "deep learning", "label": "Method", "score": 0.997}]
}
```
