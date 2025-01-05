import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from tempfile import NamedTemporaryFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import streamlit as st

# Initialize Groq client
GROQ_API_KEY="api_key"
client = Groq(api_key=GROQ_API_KEY)

# Predefined list of Google Drive links
drive_links = [
    "https://drive.google.com/file/d/1pd3XR7SBmqy12XOv0nIroL7StEsc2QMu/view?usp=sharing",
    
    # Add more links here as needed
]

# Function to download PDF from Google Drive
def download_pdf_from_drive(drive_link):
    file_id = drive_link.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(download_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception("Failed to download the PDF file from Google Drive.")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_stream):
    pdf_reader = PdfReader(pdf_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Function to create embeddings and store them in FAISS
def create_embeddings_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_db

# Function to query the vector database and interact with Groq
def query_vector_db(query, vector_db):
    # Retrieve relevant documents
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Interact with Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"Use the following context:\n{context}"},
            {"role": "user", "content": query},
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Streamlit app
st.title("RAG-Based Application with Google Drive Support for Electrical System Commissioning by SEC Laws")

st.write("Processing the predefined Google Drive links...")

all_chunks = []

# Process each predefined Google Drive link
for link in drive_links:
    try:
        st.write(f"Processing link: {link}")
        # Download PDF
        pdf_stream = download_pdf_from_drive(link)
        st.write("PDF Downloaded Successfully!")
        
        # Extract text
        text = extract_text_from_pdf(pdf_stream)
        st.write("PDF Text Extracted Successfully!")
        
        # Chunk text
        chunks = chunk_text(text)
        st.write(f"Created {len(chunks)} text chunks.")
        all_chunks.extend(chunks)
    except Exception as e:
        st.write(f"Error processing link {link}: {e}")

if all_chunks:
    # Generate embeddings and store in FAISS
    vector_db = create_embeddings_and_store(all_chunks)
    st.write("Embeddings Generated and Stored Successfully!")
    
    # User query input
    user_query = st.text_input("Enter your query:")
    if user_query:
        response = query_vector_db(user_query, vector_db)
        st.write("Response from LLM:")
        st.write(response)
