from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone import Pinecone
from dotenv import load_dotenv
import hashlib
import os
from src.helper import (
    load_pdf_files,
    filter_to_minimal_documents,
    text_splitter,
    download_embeddings,
)


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# --------------------------------------------------
# Configuration
# --------------------------------------------------

#INDEX_NAME = "medical-chatbot-st-all-minilm"
INDEX_NAME = "medical-chatbot-gemini"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        #dimension=384,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        ),
        )
index = pc.Index(INDEX_NAME)


# Folder containing ONLY new PDFs
DATA_PATH = "data"

# --------------------------------------------------
# Load Documents
# --------------------------------------------------
print("Loading PDFs...")

documents = load_pdf_files(DATA_PATH)

if not documents:
    print("No PDF files found.")
    exit()

minimal_docs = filter_to_minimal_documents(documents)

chunks = text_splitter(minimal_docs)

print(f"Total chunks created: {len(chunks)}")

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = download_embeddings()

# --------------------------------------------------
# Connect to Existing Pinecone Index
# --------------------------------------------------

vector_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

# --------------------------------------------------
# Generate Unique IDs
# --------------------------------------------------

ids = []

for chunk in chunks:

    source = chunk.metadata.get("source", "")

    unique_text = source + chunk.page_content

    doc_id = hashlib.md5(
        unique_text.encode("utf-8")
    ).hexdigest()

    ids.append(doc_id)

# --------------------------------------------------
# Add Documents
# --------------------------------------------------

print("Uploading embeddings to Pinecone...")

vector_store.add_documents(
    documents=chunks,
    ids=ids,
)

print("Knowledge base updated successfully.")
print(f"Added {len(chunks)} chunks.")