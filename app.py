from flask import Flask, render_template, request, session
from src.helper import rag_chain, download_embeddings, get_llm
from src.logger_config import setup_logger
import uuid

from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os

logger = setup_logger()
app = Flask(__name__)
app.secret_key = "medical-chatbot-secret"
# --------------------------------------------------
# Load environment variables from .env file 
# --------------------------------------------------

retriever = None
llm = None

def initialize():

    global retriever, llm

    try:
        logger.info("Loading environment variables")
        load_dotenv()
        
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY missing")

        logger.info("Loading embeddings")
        embeddings = download_embeddings()

        logger.info("Connecting Pinecone")
        docsearch = PineconeVectorStore.from_existing_index(index_name="medical-chatbot-st-all-minilm",embedding=embeddings)

        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        logger.info("Initializing Groq & Gimini")
        groq = get_llm("groq")
        gemini = get_llm("gemini")

        logger.info("Configuring fallback LLM")
        llm = groq.with_fallbacks([gemini])
        
        logger.info("Application initialized successfully")

    except Exception:
        logger.exception("Application startup failed")
        raise
    

################################################################################################
@app.route("/")
def index():
    try:
        return render_template("chat.html")
    except Exception:
        logger.exception("Failed loading home page")
        return "Application Error", 500


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        session_id = session["session_id"]
        
        msg = request.form["msg"]
        logger.info(f"Question: {msg}")
        logger.info(f"Session ID: {session_id}")
        response = rag_chain(msg, retriever, llm, session_id)
        logger.info("Response generated successfully")
        return str(response)
    
    except Exception:
        logger.exception("Chat request failed")
        return "Sorry, an internal error occurred."

if __name__ == "__main__":
    try:
        initialize()

        logger.info("=" * 60)
        logger.info("Medical Chatbot Started Successfully")
        logger.info("Local URL  : http://127.0.0.1:8080")
        logger.info("Network URL: http://localhost:8080")
        logger.info("=" * 60)
        
        app.run(host="0.0.0.0", port=8080, debug=False)

    except Exception:
        logger.exception("Application crashed during startup")


