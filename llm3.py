import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import OpenAI
import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Function to read documents
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Function to chunk the documents
def chunk_data(docs, chunk_size=300, chunk_overlap=20):
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

# Store embeddings in Chroma DB
def store_embeddings_in_chroma(embeddings, chunks, db_dir="chroma_db"):
    chroma_store = Chroma(embedding_function=get_embedding_function(), persist_directory=db_dir)
    chroma_store.add_documents(documents=chunks, embeddings=embeddings)
    chroma_store.persist()  # Save the embeddings and metadata to the disk
    print(f"Embeddings stored in ChromaDB at {db_dir}")
# Function to run a query
def run_query(query, chroma_store):
    chroma_store = Chroma(embedding_function=get_embedding_function(), persist_directory=db_dir)
    
    # Convert the prompt into embeddings
    embedding_model = get_embedding_function()
    prompt_embedding = embedding_model.embed_query(prompt)
    # Get embedding for the query

    # Search for similar documents in the Chroma DB
    results = chroma_store.similarity_search_by_vector(prompt_embedding,k=1) # Retrieve top 5 results
    return results

# Main logic
if __name__ == "__main__":
    # Load documents and process them
    doc = read_doc('documents/')
    print(f"Number of documents: {len(doc)}")
    
    documents = chunk_data(docs=doc)  # Chunk the documents
    print(f"Number of chunks: {len(documents)}")

    # Get embeddings for each chunk
    embeddings = embed_documents(documents)
    print(f"Generated {len(embeddings)} embeddings")

    # Store embeddings in ChromaDB
    store_embeddings_in_chroma(embeddings, documents)

    # Example query
    user_query = "What is the main topic of the document?"
    print(f"Running query: {user_query}")
    results = run_query(user_query, Chroma(embedding_function=get_embedding_function(), persist_directory="chroma_db"))
    
    # Display the results
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}:")
        print(result.page_content)  # Print the content of the retrieved document
