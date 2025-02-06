import os
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sentence_transformers import SentenceTransformer

# Download stopwords if not already present
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))


# Path to the folder
folder_path = 'cases'

# Define the clean_text function
def clean_text(text):
    """Preprocess text: remove stopwords, special characters, and lowercase."""
    # Lowercase the text
    text = text.lower()

    # Remove special characters 
    text = re.sub(r"[^a-zA-Z0-9@.:/\\\s-]", " ", text)

    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]

    # Reconstruct the cleaned text
    return " ".join(filtered_words)

# Function to create embeddings for a single text
def create_embeddings(text, embedding_model_name="all-MiniLM-L6-v2"):
    """Creates an embedding for a single text using Sentence Transformers."""
    
    # Load the model
    model = SentenceTransformer(embedding_model_name)
    
    # Generate the embedding for the provided text
    embedding = model.encode(text, convert_to_tensor=False)
    
    return embedding.tolist()

relevant_files_high = {"case_1.txt", "case_2.txt", "case_3.txt", "case_5.txt"} # Use a set for faster lookups
relevant_files_medium = "case_7.txt"
def determine_relevance(filename):
    if filename in relevant_files_high:
        return "HIGH"
    elif filename == relevant_files_medium:  # Check for the medium relevance file
        return "MEDIUM"
    else:
        return "LOW"

if __name__=='__main__':
    # Initialize an empty list to store data
    data = []

    # Loop over each file in the folder
    for file_name in os.listdir(folder_path):
        # Get the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read the content of the file and join lines into a single paragraph
        with open(file_path, 'r', encoding='utf-8') as f:
            text = " ".join(f.read().splitlines())  # Joins lines into a single paragraph

        # Clean the text using the provided function
        cleaned_text = clean_text(text)
        
        # Append the file information as a row to the data list
        data.append([file_name, text, cleaned_text])

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=['File Name', 'Text', 'Cleaned_Text'])

    # Apply the function to the 'Text' column and create a new 'Embedding' column
    df['Embedding'] = df['Cleaned_Text'].apply(create_embeddings)

    df['Relevance'] = df['File Name'].apply(determine_relevance)

    df.to_csv('document_embedding.csv')
