#!/usr/bin/env python3
"""
Semantic Document Retrieval System (RAG)
A simple Retrieval-Augmented Generation system using LangChain, OpenAI embeddings, and GPT-4
for semantically searching and reasoning over custom documents.
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize models
embeddings = OpenAIEmbeddings(api_key=os.getenv("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"))
llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=os.getenv("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"))

# Sample documents (in production, load from PDFs using PyPDF2 or similar)
sample_docs = [
    Document(page_content="Semantic search uses embeddings to find documents by meaning, not just keywords. It's the foundation of RAG systems.", metadata={"source": "rag_fundamentals.pdf"}),
    Document(page_content="Retrieval-Augmented Generation (RAG) combines retrieval of relevant documents with LLM generation for accurate, grounded answers.", metadata={"source": "rag_guide.pdf"}),
    Document(page_content="Multi-step reasoning allows agents to decompose complex questions into subtasks, retrieving context dynamically as needed.", metadata={"source": "agent_reasoning.pdf"}),
    Document(page_content="Tool calling enables LLMs to invoke functions, databases, and APIs as part of multi-step reasoning workflows.", metadata={"source": "llm_tools.pdf"}),
]

def build_vector_store(docs):
    """Build FAISS vector store from documents."""
    return FAISS.from_documents(docs, embeddings)

def create_rag_chain(vector_store):
    """Create RAG chain with custom prompt template."""
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an AI assistant with access to a knowledge base.
        
Context (retrieved documents):
{context}

Question: {question}

Provide a clear, accurate answer based on the context. If the context doesn't contain relevant information, say so."""
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

def main():
    # Build vector store
    print("Building vector store from documents...")
    vector_store = build_vector_store(sample_docs)
    
    # Create RAG chain
    print("Initializing RAG chain...")
    rag_chain = create_rag_chain(vector_store)
    
    # Example queries
    queries = [
        "What is RAG and how does it work?",
        "How do LLMs use tool calling in multi-step reasoning?",
        "Explain semantic search in the context of document retrieval.",
    ]
    
    # Run queries
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_chain.run(query)
        print(f"Answer: {response}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
