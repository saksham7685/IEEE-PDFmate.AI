
import json

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
import os
from langchain_ollama import OllamaEmbeddings

# Function to read documents
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Function to chunk the documents
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

# Function to initialize the Ollama embeddings model
def get_embedding_function():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')  # Specify the model if needed
    return embeddings

# Embed the chunks using OllamaEmbeddings
def embed_documents(chunks):
    embedding_model = get_embedding_function()  # Initialize the embeddings model
    embeddings = []
    for chunk in chunks:
        text = chunk.page_content  # Extract the text content from the chunk
        embedding = embedding_model.embed_query(text)  # Get embedding for the chunk
        embeddings.append(embedding)
    return embeddings

# Save embeddings to a JSON file
def save_embeddings(embeddings, filename="embeddings.json"):
    with open(filename, 'w') as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved to {filename}")
    

# Main logic
doc = read_doc('documents/')
print(f"Number of documents: {len(doc)}")

documents = chunk_data(docs=doc)  # Chunk the documents
print(f"Number of chunks: {len(documents)}")

# Get embeddings for each chunk
embeddings = embed_documents(documents)
print(f"Generated {len(embeddings)} embeddings")

# Save embeddings to a file
save_embeddings(embeddings)

