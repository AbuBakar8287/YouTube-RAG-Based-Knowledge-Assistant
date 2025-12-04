🎬 YouTube RAG Assistant with Chain Concept

A Streamlit app that extracts YouTube transcripts and answers user queries using RAG (Retrieval-Augmented Generation), FAISS, Cohere Reranking, and Gemini LLM.

--- Features---
-- Extract transcript

Automatically fetches the transcript using YouTube Transcript API.

-- Chunking & Vector Store

Splits transcript using RecursiveCharacterTextSplitter

Stores embeddings in FAISS vector database

Embeddings from sentence-transformers/all-MiniLM-L6-v2

-- Smart Retrieval

MMR-based Retriever for diverse chunk selection

Optional fallback if no relevant chunks found

-- Cohere Reranking

Reranks retrieved chunks using Cohere’s Reranking API for higher accuracy.

-- Gemini LLM for Answering

Uses Gemini 2.5 Flash for final answer generation

Follows strict rule: respond only from transcript context

Says “I don’t know” if transcript lacks info

--Beautiful UI

Fully styled Streamlit UI

Custom gradients, buttons, inputs, focus style, cursor visibility etc.

--Tech Stack
Component	Tool
LLM	Gemini 2.5 Flash
Reranking	Cohere Rerank
Vector DB	FAISS
Embeddings	sentence-transformers MiniLM
Frontend	Streamlit
Retrieval	MMR Retriever
Chunking	RecursiveCharacterTextSplitter




--How It Works (Architecture)
YouTube URL → Extract Video ID → Fetch Transcript → Chunk → Embeddings → FAISS Vector Store
                                   ↓
                               Retriever → Cohere Reranker → Best Chunks →
                                                                       Prompt →
                                                                        LLM →
                                                                      Final Answer



