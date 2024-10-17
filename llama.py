from sentence_transformers import SentenceTransformer
import numpy as np
def text_embeddings(text)->None:
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text,normalise_embeddings=True)
phrase="Appleeeee is a fruit"
embedding1= text_embeddings(phrase)
print(len(embedding1))
