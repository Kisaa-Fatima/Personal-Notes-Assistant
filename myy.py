import os
import PyPDF2
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Create necessary directories
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

# Load LLM using Groq's ChatGroq
def load_llm():
    groq_api_key = "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"
    model_name = "Llama3-8b-8192"  # Specify the Groq-hosted model
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    # Use the uploaded file's buffer directly
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Preprocess the extracted text
def preprocess_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Build or load the vector database
def build_or_load_vectordb(chunks, embeddings, db_path="vectorstore/db_chroma"):
    if os.path.exists(db_path):
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        db = Chroma.from_texts(chunks, embeddings, persist_directory=db_path)
        db.persist()
    return db

# Define the retrieval-based QA system
def create_retrieval_qa_system(vector_db, llm):
    prompt = PromptTemplate(
        template="""You are an assistant. Use the context below to answer the question:
        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=["context", "question"],
    )
    qa_system = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_system

# Streamlit app
def main():
    st.title("PDF Question-Answer System")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            document_text = extract_text_from_pdf(uploaded_file)
            st.success("PDF text extracted successfully!")

        # Preprocess text
        with st.spinner("Splitting text into chunks..."):
            text_chunks = preprocess_text(document_text)
            st.success("Text split into chunks successfully!")

        # Load embeddings and build/load vector database
        with st.spinner("Loading embeddings and building/loading vector database..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_db = build_or_load_vectordb(text_chunks, embeddings)
            st.success("Vector database ready!")

        # Load LLM
        with st.spinner("Loading LLM..."):
            llm = load_llm()
            st.success("LLM loaded successfully!")

        # Create QA system
        with st.spinner("Setting up QA system..."):
            qa_system = create_retrieval_qa_system(vector_db, llm)
            st.success("QA system is ready!")

        # User query
        st.header("Ask a Question About the PDF")
        user_query = st.text_input("Your question:")
        if user_query:
            with st.spinner("Searching for an answer..."):
                response = qa_system.run(user_query)
                st.success("Answer generated!")
                st.write(f"**Answer:** {response}")

if __name__ == "__main__":
    main()
