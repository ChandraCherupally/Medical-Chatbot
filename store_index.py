from src.helper import load_pdf_files, filter_to_minimal_documents, text_splitter, download_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

## 1. Load environment variables from .env file
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.environ.get('PINECONE_API_KEY')
os.environ["GEMINI_API_KEY"] = os.environ.get('GEMINI_API_KEY')

## 2. Load PDF documents and split them into smaller text chunks
extracted_data=load_pdf_files(data='data/')
filter_data = filter_to_minimal_documents(extracted_data)
text_chunks=text_splitter(filter_data)

## 3. Initialize the embedding model (all-MiniLM-L6-v2)
embeddings = download_embeddings()

## 4. Connect to Pinecone and create the vector index if it does not exist
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = "medical-chatbot-st-all-minilm"  # change if desired
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

## 5. Connect to the Pinecone index
index = pc.Index(index_name)


## 6. Generate vector embeddings and upsert them into Pinecone
vector_store = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
