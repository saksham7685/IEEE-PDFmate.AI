import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO
from transformers import pipeline

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
    
    return summary

# Provide the path to your PDF
pdf_path = "demo.pdf"  # Adjust this if the file is not in the same directory
summary = summarize_pdf(pdf_path, chunk_size=1000, max_length=150, min_length=50)
print("Summary of the PDF:")
print(summary)