import os
import ast
from typing import List, Optional, Union

from langchain.callbacks import FileCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from loguru import logger
from rich import print
from sentence_transformers import CrossEncoder
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

logfile = "log/output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)


persist_directory = None


class RAGException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def load_embedding_model(
        model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu"
    ) -> HuggingFaceBgeEmbeddings:
        """
        Loads and returns a HuggingFaceBgeEmbeddings model for embedding text. This embedding model is used to encode the documents and queries for retrieval.

        Args:
            model_name (str): The name or path of the pre-trained model to load. Defaults to "BAAI/bge-large-en-v1.5".
            device (str): The device to use for model inference. Defaults to "cuda".

        Returns:
            HuggingFaceBgeEmbeddings: The loaded HuggingFaceBgeEmbeddings model.
        """
        model_kwargs = {"device": device}
        encode_kwargs = {
            "normalize_embeddings": True
        }  # set True to compute cosine similarity
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embedding_model

    def generate_repo_ast(repo_path):
    repo_summary = {}
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        # Count the number of each type of AST node
                        node_counts = {}
                        for node in ast.walk(tree):
                            node_type = type(node).__name__
                            node_counts[node_type] = node_counts.get(node_type, 0) + 1
                        repo_summary[file_path] = node_counts
                    except SyntaxError:
                        repo_summary[file_path] = "Unable to parse file"
    return repo_summary