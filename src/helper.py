from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from src.prompt import *
from src.logger_config import setup_logger
from langchain_openai import ChatOpenAI
import os

from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict, total=False):
    question: str
    context: str
    answer: str
    chat_history: List[list]

logger = setup_logger()

memory = MemorySaver()
graph = None

#1. Extract Data From the PDF File into docs
def load_pdf_files(data) -> List[Document]:
    logger.info(f"Loading PDFs from {data}")
    loader = DirectoryLoader(data, glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} PDF files")
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
    logger.info(f"Splitting {len(minimal_docs)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, length_function=len)
    split_docs = text_splitter.split_documents(minimal_docs)
    logger.info(f"Created {len(split_docs)} chunks")
    return split_docs


#4. Download the Embeddings from HuggingFace 
def download_embeddings():
    "Downloads the embeddings from HuggingFace and returns the embeddings object."
    logger.info("Loading embeddings model")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    logger.info("Embeddings loaded successfully")
    return embeddings


def get_llm(provider: str):

    logger.info(f"Creating LLM: {provider}")

    if provider == "groq":        
        return ChatOpenAI(api_key=os.getenv("GROQ_API_KEY"), 
                          base_url="https://api.groq.com/openai/v1",
                          model="llama-3.3-70b-versatile",  ##llama3-8b-8192 - decomissioned
                          temperature=0)

    elif provider == "gemini":
        return ChatOpenAI(api_key=os.getenv("GEMINI_API_KEY"), 
                          base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
                          model="gemini-2.5-flash",
                          temperature=0)

    else:
        raise ValueError(f"Unsupported provider: {provider}")




def build_graph(llm):

    logger.info("Building LangGraph workflow")

    def chatbot_node(state: ChatState):        
        logger.info("Executing chatbot node")        
        chain = (prompt | llm | StrOutputParser())

        history = state.get("chat_history", [])
        history.append({"role": "user", "content": state["question"]})
        formatted_history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)

        logger.info(f"Question: {state['question']}")
        logger.info(f"History Length: {len(history)}")
        logger.info(f"Retrieved Context Length: {len(state['context'])}")
        logger.info(f"History Contents: {formatted_history}")
        response = chain.invoke({"chat_history": formatted_history, 
                                 "context": state["context"],
                                 "question": state["question"]})

        logger.info("Response generated successfully")

        history.append({"role": "assistant", "content": response})
        return {"answer": response, "chat_history": history}

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=memory)




#5. Creating RAG pipeline
def rag_chain(question, retriever, llm, session_id):

    global graph

    try:

        logger.info(f"Retrieving documents for session {session_id}")
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} chunks")

        context = "\n\n".join(doc.page_content for doc in docs)

        if graph is None:
            logger.info("Initializing LangGraph")
            graph = build_graph(llm)

        result = graph.invoke({"question": question, "context": context},
                              config={"configurable": {"thread_id": session_id}})

        logger.info("LangGraph response generated")

        return result["answer"]

    except Exception:
        logger.exception("RAG pipeline failed")
        raise
        