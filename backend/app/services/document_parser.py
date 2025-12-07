# apps/backend/app/services/document_parser.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

def parse_file(file_path: str, file_type: str) -> List[Document]:
    """
    Parses a file based on its type and returns a list of Document chunks.
    """
    file_type = file_type.lower().strip('.')
    
    loader_map = {
        'pdf': PyPDFLoader,
        'docx': Docx2txtLoader,
        'doc': Docx2txtLoader,
        'txt': TextLoader,
        'csv': CSVLoader,
    }

    loader_class = loader_map.get(file_type)

    if loader_class:
        if file_type in ['txt', 'csv']:
            loader = loader_class(file_path, encoding='utf-8')
        else:
            loader = loader_class(file_path)
    else:
        # Fallback for other types like .xlsx, .pptx, etc.
        loader = UnstructuredFileLoader(file_path)

    documents = loader.load()

    # Split documents into smaller chunks for better RAG performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    return text_splitter.split_documents(documents)