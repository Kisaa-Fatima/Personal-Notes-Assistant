#correct backend 
import os
import PyPDF2
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
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
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

# Main process
def main():
    # Step 1: Upload and extract text
    pdf_path = input("Enter the path to your PDF file: ")
    if not os.path.exists(pdf_path):
        print("Error: File not found.")
        return
    
    print("Extracting text from PDF...")
    document_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Preprocess text and generate embeddings
    print("Splitting text into chunks...")
    text_chunks = preprocess_text(document_text)
    
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Building/loading vector database...")
    vector_db = build_or_load_vectordb(text_chunks, embeddings)
    
    # Step 3: Load the LLM
    print("Loading LLM...")
    llm = load_llm()
    
    # Step 4: Create the QA system
    print("Setting up the QA system...")
    qa_system = create_retrieval_qa_system(vector_db, llm)
    
    # Step 5: Answer user queries
    print("\nThe system is ready! Ask questions related to the uploaded PDF.")
    while True:
        user_query = input("\nYour question (type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        
        print("Searching for an answer...")
        response = qa_system.run(user_query)
        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()
