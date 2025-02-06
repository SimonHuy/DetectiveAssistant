# DetectiveAssistant

AI-Powered Retrieval-Augmented Generation (RAG) System

📌 Overview
This project implements a Retrieval-Augmented Generation (RAG) system that utilizes LanceDB for vector storage, a Hugging Face model for embeddings, and cosine similarity for efficient document retrieval. It enables users to query a dataset of documents and retrieve the most relevant results based on similarity.

⚙️ Features
+ Embeddings Generation: Uses a Hugging Face model to generate embeddings for text.
+ Vector Database: Stores embeddings in LanceDB for efficient retrieval.
+ Cosine Similarity Search: Retrieves the most relevant documents using cosine similarity.
+ Filtering & Sorting: Filters documents based on relevance and sorts by similarity score.

🛠️ Installation
Prerequisites
Ensure you have Python 3.12.8 installed along with the required dependencies:

```python
pip install -r requirements.txt
```
🚀 Usage

1️⃣ Prepare the Document Embeddings
Place your documents in document_embedding.csv and ensure they have the following columns:

File Name - The name of the document.
Text - The document content.
Cleaned_Text - The cleaned document content.
Embedding - The precomputed embedding (as a list).
Relevance - Labels (HIGH, MEDIUM, LOW).

Or you can run
```
python preprocess_embed.py
```
2️⃣ Query, Retrieve and Display Results via UI
```
streamlit run UI.py
