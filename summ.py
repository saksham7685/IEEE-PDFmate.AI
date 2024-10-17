import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
from transformers import T5Tokenizer, pipeline

import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from the PDF, including text from images and tables
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            # Extract regular text
            pdf_text += page.extract_text() or ""
            
            # Extract tables and add to the text
            tables = page.extract_tables()
            for table in tables:
                pdf_text += "\n" + " | ".join([" | ".join(row) for row in table if row]) + "\n"
            
            # Extract images and run OCR
            for img in page.images:
                try:
                    # Get the image as a byte array
                    x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                    image = page.within_bbox((x0, top, x1, bottom)).to_image()

                    # Extract the original image as a PIL image
                    pil_image = image.original
                    
                    # Convert PIL Image to bytes and apply OCR
                    img_byte_arr = BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')  # Save to bytes
                    img_byte_arr.seek(0)  # Move to the beginning of the BytesIO buffer
                    ocr_text = pytesseract.image_to_string(Image.open(img_byte_arr))
                    pdf_text += "\n[Image Text]: " + ocr_text + "\n"
                except Exception as e:
                    print(f"Error processing image: {e}")
    return pdf_text

# Function to summarize long text in chunks
def summarize_long_text(text, chunk_size=1000, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Split text into chunks of chunk_size
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Summarize each chunk
    summaries = []
    for chunk in text_chunks:
        # Adjust max_length dynamically
        input_length = len(chunk)
        adjusted_max_length = min(max_length, input_length // 2)
        summary = summarizer(chunk, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    # Combine summaries
    full_summary = " ".join(summaries)
    return full_summary

# Main function to extract and summarize the PDF content
def summarize_pdf(pdf_path, chunk_size=1000, max_length=150, min_length=50):
    # Step 1: Extract text, tables, and images (with OCR) from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Summarize the extracted content
    summary = summarize_long_text(pdf_text, chunk_size=chunk_size, max_length=max_length, min_length=min_length)
    
    return pdf_text, summary

# Function to extract topics from the text using LDA
def extract_topics(text, num_topics=5):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    dictionary = corpora.Dictionary([filtered_words])
    doc_term_matrix = [dictionary.doc2bow(filtered_words)]

    lda_model = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)
    topics = lda_model.print_topics(num_words=4)
    return topics

# Function to generate questions from the text
def generate_questions(text, num_questions=5, max_length=128):
    tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl", use_fast=True)
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl", num_beams=5)

    questions = []
    chunk_size = 512  # Adjust based on model limits
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        input_text = f"generate questions: {chunk}"

        try:
            generated_questions = question_generator(input_text, max_length=max_length, num_return_sequences=num_questions)
            questions.extend([q['generated_text'] for q in generated_questions])
        except Exception as e:
            print(f"Error generating questions for chunk: {chunk}\nError: {str(e)}")

    return questions

# Provide the path to your PDF
pdf_path = "C:/Users/nisht/OneDrive/Desktop/default Project/documents/Summer Intern Hiring_PP and DE_Btech Interns.pdf"

# Extract and summarize the text from the PDF
extracted_text, summary = summarize_pdf(pdf_path, chunk_size=1000, max_length=150, min_length=50)

# Print the summary of the PDF
print("Summary of the PDF:")
print(summary)

# Generate and print questions from the extracted text
questions = generate_questions(extracted_text, num_questions=5)

print("\nGenerated Questions:")
for i, question in enumerate(questions, 1):
    print(f"Question {i}: {question}")
