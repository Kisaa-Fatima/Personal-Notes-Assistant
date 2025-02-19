# Notes-Personal-Assistant

You somethimes find difficulty in remembering what you wrote in your notes or what that term was in the book, or what were the examples related to that specefic topic!? Dont worry your personal chat assistant is here! Upload a book of 600 pages or your semester notes, you can get any answer you want from your notes, now no need to find youself and read everything because using this you can get a quick answer to what you are looking for. 

This repository contains your very own Personal notes based Q/Answer System built with Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Groq’s LLM. The application enables users to upload notes as PDF documents, extract text, create a vector-based retrieval system, and perform question-answering using LLM-powered retrieval-augmented generation (RAG).

## Features
PDF Parsing: Extracts text from uploaded PDFs using PyPDF2.
Text Preprocessing: Utilizes RecursiveCharacterTextSplitter for chunking text to enhance retrieval efficiency.
Vector Database: Implements Chroma as a persistent vector storage solution.
Hugging Face Embeddings: Leverages sentence-transformers/all-MiniLM-L6-v2 to create dense vector representations of document chunks.
LLM Integration: Uses Groq-hosted Llama3-8b-8192 to generate answers based on retrieved document context.
Streamlit UI: Provides an interactive and user-friendly interface for seamless PDF Q&A interactions.

## Tech Stack
Python
Streamlit
LangChain
ChromaDB
Hugging Face Transformers
Groq API (Llama3-8b-8192)


