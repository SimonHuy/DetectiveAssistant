import pandas as pd
import ast
import pyarrow as pa
import lancedb
import os
import shutil
import ast
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

gemini_api = os.getenv("GEMINI_API")

from embedding import create_embeddings # Assuming embedding.py defines create_embeddings

EMEBDDING_FILE = 'document_embedding.csv'
URI = 'data'
if os.path.exists(URI) and os.path.isdir(URI):
    shutil.rmtree(URI)
    print('remove the dbdata folder')
else:
    print('create datadb folder')
db = lancedb.connect(URI)

def read_embedding():
    df = pd.read_csv(EMEBDDING_FILE)
    df['Embedding'] = df['Embedding'].apply(lambda x: ast.literal_eval(x))
    
    return df 

def save_embedding():
    df = read_embedding()
    custom_schemma = pa.schema(
        [
            pa.field("File Name", pa.string()),
            pa.field("Text", pa.string()),
            pa.field("Embedding", pa.list_(pa.float64(),384)),
            pa.field("Relevance", pa.string()),
        ]
    )
    db.create_table('document_table', data=df, schema=custom_schemma)

save_embedding()
TABLE_NAME = 'document_table'


def search_documents(query, top_k=5):
    """
    Searches the LanceDB database for documents similar to the given query.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to return.  Defaults to 5.

    Returns:
        pandas.DataFrame: A DataFrame containing the top_k most similar documents.
    """
    table = db.open_table(TABLE_NAME)
    query_embedding = create_embeddings(query) #  Use the embedding function to embed the query
    results = table.search(query_embedding).metric("cosine").to_df()
    results['Cosine_Score'] = 1 - results['_distance']
    results = results[['File Name', 'Text', 'Relevance', 'Cosine_Score']]
    results = results[(results['Relevance'] == 'HIGH') | (results['Relevance'] == 'MEDIUM')]
    results = results.sort_values(by='Cosine_Score', ascending=False)
    return results.head(top_k)

st.title('AI Detective Challenge: Retrieval-Augmented Generation (RAG) Test')

q1 = st.text_input('Query')

if q1:
    result = search_documents(q1).iloc[0]
    
    result_text = result['Text']
    st.write('Most matched Document')
    st.write(result)
    genai.configure(api_key=gemini_api)

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        system_instruction=f"You are an AI that assist detectives in solving a cybercrime case. Use the provide information to answer the detective query:\n\nInformation: {result_text}\n\n",
        )

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            q1,
        ],
        }
    ]
    )

    response = chat_session.send_message("INSERT_INPUT_HERE")

    st.write('Summary Response')
    st.write(response.text)