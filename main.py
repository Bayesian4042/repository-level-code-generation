import os
import json

os.environ["HF_HOME"] = "./weights"
os.environ["TORCH_HOME"] = "./weights"

import gc
import re
import uuid
import subprocess
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext

from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding



from rag.retriever import (
    load_embedding_model,
    generate_repo_ast
)

# Initialize the 'id' attribute if it doesn't exist
if 'id' not in st.session_state:
    st.session_state.id = uuid.uuid4()  # or set to some default value

# Initialize chat history and query engine if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# Setting up the llm
llm = Ollama(model="llama3", request_timeout=120.0, base_url="http://localhost:11434")

# Setting up the embedding model
lc_embedding_model = load_embedding_model()
embed_model = LangchainEmbedding(lc_embedding_model)

# Utility functions
def initialize_query_engine():
    if "query_engine" not in st.session_state or st.session_state.query_engine is None:
        st.error("Query engine is not initialized. Please load a repository first.")
        return False
    return True

def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def clone_repo(repo_url):
    return subprocess.run(["git", "clone", repo_url], check=True, text=True, capture_output=True)

def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

with st.sidebar:
    # Input for GitHub URL
    github_url = st.text_input("GitHub Repository URL")

    # Button to load and process the GitHub repository
    process_button = st.button("Load")

    message_container = st.empty()  # Placeholder for dynamic messages

    if process_button and github_url:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            with st.spinner(f"Loading {repo} repository by {owner}..."):
                try:
                    input_dir_path = f"./full-stack-nextjs"
                    
                    if not os.path.exists(input_dir_path):
                        subprocess.run(["git", "clone", github_url], check=True, text=True, capture_output=True)

                    if os.path.exists(input_dir_path):
                        loader = SimpleDirectoryReader(
                            input_dir=input_dir_path,
                            recursive=True
                        )
                    else:    
                        st.error('Error occurred while cloning the repository, carefully check the url')
                        st.stop()

                    docs = loader.load_data()

                    # ====== Create vector store and upload data ======
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs)

                    # ====== Setup a query engine ======
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming_response=True, similarity_top_k=4)
                    
                    # ====== Customize prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    """Answer the user's coding questions with code and file name as well. Make sure all the imports are present and the code is the one file in a continous manner.
                    Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n
                    "Query: {query_str}\n"""
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    if docs:
                        message_container.success("Data loaded successfully!!")
                    else:
                        message_container.write(
                            "No data found, check if the repository is not empty!"
                        )
                    st.session_state.query_engine = query_engine
                    st.session_state.repo_ast = generate_repo_ast(f"./full-stack-nextjs")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()

                st.success("Ready to Chat!")
        else:
            st.error('Invalid owner or repository')
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat With Your Code")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    if not initialize_query_engine():
        st.stop()
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # context = st.session_state.context
        query_engine = st.session_state.query_engine

        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(f'Given the repository AST:\n{json.dumps(st.session_state.repo_ast, indent=2)}\n\nAnd the following question: {prompt}')
        print(streaming_response)
        # for chunk in streaming_response.response_gen:
        #     full_response += chunk
        #     message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(streaming_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
