## Install uv
- powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

## Install Python Through uv
- uv python install 3.12

## Create Project
- mkdir my-ai-project
- cd my-ai-project

## Create Virtual Environment and Activate
- uv venv 
- .venv\Scripts\activate

## Install Packages
- uv pip install pandas openai langchain

## Install pip into the environment for better VSCode compatibility.
- uv pip install pip

## Create requirements.txt file and installing
- uv pip freeze > requirements.txt
- uv pip install -r requirements.txt

## Project creation work flow
- uv init my-ai-project
- cd my-ai-project
- uv venv --python 3.12
- uv init
- .venv\Scripts\activate
- uv add jupyter ipykernel
- uv add gdown
- uv add pandas
- uv add scikit-learn requests
- uv add matplotlib seaborn
- uv add nltk wordcloud
- uv add gensim spacy

project/
 ├── .venv/  --> Actual installed environment 
 ├── pyproject.toml  -->What project NEEDS
 ├── uv.lock --> Exact versions project USES
 └── main.py

## Folder structure
Folder          = normal directory
__init__.py     = "this directory is a Python package"
Package         = reusable Python module collection

## Real AI Project Example

src/
└── intellisupport_agent/
    ├── __init__.py
    ├── api/
    │   ├── __init__.py
    │   └── server.py
    ├── rag/
    │   ├── __init__.py
    │   └── retriever.py
    └── utils/
        ├── __init__.py
        └── helpers.py

## Now we can import
- from intellisupport_agent.rag.retriever import retrieve        
