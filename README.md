# Semantic-Document-Retrieval-System-RAG-

# Semantic Document Retrieval System (RAG)

This project implements a simple **Retrieval-Augmented Generation (RAG)** pipeline that performs semantic search over a small set of documents using vector embeddings and a language model. It showcases the core building blocks behind modern AI agents: embeddings, retrieval, and LLM-based reasoning.

---

## Features

- Semantic search over documents using vector embeddings and FAISS
- Retrieval-Augmented Generation (RAG): retrieve context → answer with an LLM
- Custom prompt template for grounded, context-aware answers
- Clear, minimal code that is easy to extend with real documents (e.g., PDFs)

---

## Project Structure

- `rag_system.py`  
  Main script that:
  - Defines example documents
  - Builds a FAISS vector store from embeddings
  - Creates a RAG-style QA chain
  - Runs several example queries and prints the answers

---
