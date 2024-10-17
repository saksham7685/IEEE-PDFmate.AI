from flask import Flask, request, jsonify
import os
import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
from transformers import T5Tokenizer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            pdf_text += page.extract_text() or ""
            tables = page.extract_tables()
            for table in tables:
                pdf_text += "\n" + " | ".join([" | ".join(row) for row in table if row]) + "\n"
            for img in page.images:
                try:
                    x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                    image = page.within_bbox((x0, top, x1, bottom)).to_image()
                    pil_image = image.original
                    img_byte_arr = BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    ocr_text = pytesseract.image_to_string(Image.open(img_byte_arr))
                    pdf_text += "\n[Image Text]: " + ocr_text + "\n"
                except Exception as e:
                    print(f"Error processing image: {e}")
    return pdf_text

# Function to summarize long text
def summarize_long_text(text, chunk_size=1000, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in text_chunks:
        input_length = len(chunk)
        adjusted_max_length = min(max_length, input_length // 2)
        summary = summarizer(chunk, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    full_summary = " ".join(summaries)
    return full_summary

# Function to summarize PDF
def summarize_pdf(pdf_path, chunk_size=1000, max_length=150, min_length=50):
    pdf_text = extract_text_from_pdf(pdf_path)
    summary = summarize_long_text(pdf_text, chunk_size=chunk_size, max_length=max_length, min_length=min_length)
    return pdf_text, summary

# Function to read documents
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Function to chunk the documents
# Function to chunk the documents
def chunk_data(docs, chunk_size=500, chunk_overlap=20):
    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(docs)
    
    # Convert chunks to the desired dictionary format
    formatted_chunks = [{'page_content': chunk.page_content} for chunk in chunks]
    
    return formatted_chunks


# Function to get embedding model
def get_embedding_function():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings

# Function to embed documents
def embed_documents(chunks):
    embedding_model = get_embedding_function()
    embeddings = []
    for chunk in chunks:
        text = chunk.page_content
        embedding = embedding_model.embed_query(text)
        embeddings.append(embedding)
    return embeddings

# Store embeddings in Chroma DB
def store_embeddings_in_chroma(embeddings, chunks, db_dir="chroma_db"):
    chroma_store = Chroma(embedding_function=get_embedding_function(), persist_directory=db_dir)
    chroma_store.add_documents(documents=chunks, embeddings=embeddings)
    chroma_store.persist()
    print(f"Embeddings stored in ChromaDB at {db_dir}")

# Query ChromaDB using a user prompt
def query_uploaded_document(prompt, db_dir="chroma_db"):
    chroma_store = Chroma(embedding_function=get_embedding_function(), persist_directory=db_dir)
    embedding_model = get_embedding_function()
    prompt_embedding = embedding_model.embed_query(prompt)
    results = chroma_store.similarity_search_by_vector(prompt_embedding, k=1)
    return results

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     pdf_path = f"uploads/{file.filename}"
#     file.save(pdf_path)

#     extracted_text, summary = summarize_pdf(pdf_path)
    
#     # Generate embeddings and store in Chroma
#     documents = chunk_data([{'page_content': extracted_text}])
#     embeddings = embed_documents(documents)
#     store_embeddings_in_chroma(embeddings, documents)

#     return jsonify({
#         'summary': summary,
#         'extracted_text': extracted_text
#     })

# @app.route('/query', methods=['POST'])
# def query():
#     data = request.get_json()
#     prompt = data.get('prompt', '')
#     results = query_uploaded_document(prompt)
#     return jsonify({
#         'results': [result.page_content for result in results]
#     })

# if __name__ == '__main__':
#     os.makedirs('uploads', exist_ok=True)
#     app.run(debug=True)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Specify the custom path where you want to save the uploaded PDF
    pdf_path = f"my_pdfs/{file.filename}"  # Change 'my_pdfs' to your desired directory
    file.save(pdf_path)

    extracted_text, summary = summarize_pdf(pdf_path)
    
    # Generate embeddings and store in Chroma
    documents = chunk_data([{'page_content': extracted_text}])
    embeddings = embed_documents(documents)
    store_embeddings_in_chroma(embeddings, documents)

    return jsonify({
        'summary': summary,
        'extracted_text': extracted_text
    })

if __name__ == '__main__':
    os.makedirs('my_pdfs', exist_ok=True)  # Ensure the custom directory exists
    app.run(debug=True)
