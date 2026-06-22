from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from langchain_core.output_parsers import StrOutputParser
from prompt import *

#1. Extract Data From the PDF File
def load_pdf_files(data) -> List[Document]:
    loader = DirectoryLoader(data, glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


#2. Filtering only content and source_metadata from document
def filter_to_minimal_documents(docs: List[Document]) -> List[Document]:
    minimal_docs : List[Document] = []
    for doc in docs:
        minimal_doc = Document(page_content=doc.page_content,metadata={"source": doc.metadata.get("source", "unknown")})
        minimal_docs.append(minimal_doc)
    return minimal_docs


#3. Split the Data into Text Chunks
def text_splitter(minimal_docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, length_function=len)
    split_docs = text_splitter.split_documents(minimal_docs)
    return split_docs

#4. Download the Embeddings from HuggingFace 
def download_embeddings():
    "Downloads the embeddings from HuggingFace and returns the embeddings object."
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


#5. Creating RAG pipeline
def rag_chain(question, retriever, llm) -> str:    

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = (prompt | llm | StrOutputParser())
    response = chain.invoke(
        {
            "context": context,
            "question": question
            }
            )
    return response
