from flask import Flask, render_template, jsonify, request
from src.helper import rag_chain, download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

## 0. Load environment variables from .env file
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.environ.get('PINECONE_API_KEY')
os.environ["GEMINI_API_KEY"] = os.environ.get('GEMINI_API_KEY')

embeddings = download_embeddings()
index_name = "medical-chatbot-st-all-minilm" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
llm = ChatOpenAI(api_key=os.environ['GEMINI_API_KEY'], 
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
                 model="gemini-2.5-flash")


######################################################
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain(msg, retriever, llm)
    print("Response : ", response)
    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080, debug=True)

