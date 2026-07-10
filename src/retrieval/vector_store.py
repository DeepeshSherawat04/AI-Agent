"""
Vector store and retrieval system for GigaCorp Support Agent.
Handles document loading, chunking, embedding, and FAISS-based retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Cached HuggingFace embeddings to avoid reloading model on every restart.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


class GigaCorpVectorStore:
    """
    Manages the FAISS vector store for GigaCorp FAQ documents.
    Uses local HuggingFace embeddings (free, no API key needed).
    """
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_store_path = Path(settings.vector_store_path)
        self.vector_store: Optional[FAISS] = None
        
    def load_documents(self, file_path: str = "data/gigacorp_faq.md") -> List[Document]:
        """
        Load and split the FAQ markdown document into chunks.
        Uses MarkdownHeaderTextSplitter for semantic chunking.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"FAQ file not found: {file_path.absolute()}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
        
        # Split by headers first for semantic chunks
        headers_to_split_on = [
            ("#", "category"),
            ("##", "section"),
            ("###", "question"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        docs = markdown_splitter.split_text(markdown_text)
        
        # Further split large chunks if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        final_docs = []
        for doc in docs:
            if len(doc.page_content) > 800:
                splits = text_splitter.split_documents([doc])
                final_docs.extend(splits)
            else:
                final_docs.append(doc)
        
        # Add metadata for source citation
        for i, doc in enumerate(final_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["source_file"] = str(file_path)
            # Extract source line reference if present in content
            if "**Source:**" in doc.page_content:
                source_line = doc.page_content.split("**Source:**")[-1].split("\n")[0].strip()
                doc.metadata["source_citation"] = source_line
        
        print(f"📄 Loaded {len(final_docs)} document chunks from FAQ")
        return final_docs
    
    def build_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Build FAISS vector store from documents.
        """
        print("🔨 Building FAISS vector store with local embeddings...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Save to disk for persistence
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_path))
        print(f"💾 Vector store saved to {self.vector_store_path}")
        
        return self.vector_store
    
    def load_existing_store(self) -> Optional[FAISS]:
        """
        Load existing FAISS vector store from disk.
        """
        if not self.vector_store_path.exists():
            return None
            
        print(f"📂 Loading existing vector store from {self.vector_store_path}")
        self.vector_store = FAISS.load_local(
            folder_path=str(self.vector_store_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vector_store
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get the retriever interface for the vector store.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() or load_existing_store() first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": settings.retrieval_k}
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search and return documents with scores.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.similarity_search_with_score(query, k=k)


def initialize_vector_store(force_rebuild: bool = False) -> GigaCorpVectorStore:
    """
    Initialize the vector store. Loads existing if available, otherwise builds from FAQ.
    """
    store = GigaCorpVectorStore()
    
    if not force_rebuild:
        existing = store.load_existing_store()
        if existing is not None:
            return store
    
    documents = store.load_documents()
    store.build_vector_store(documents)
    return store