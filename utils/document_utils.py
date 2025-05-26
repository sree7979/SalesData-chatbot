"""
Document utilities for processing and retrieving documents for RAG.
"""
import os
from typing import List, Dict, Any, Optional
import logging

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class DocumentProcessor:
    """
    A class to process documents for RAG.
    """
    def __init__(self, docs_dir: str, db_dir: str = "chroma_db"):
        """
        Initialize the document processor.
        
        Args:
            docs_dir: Directory containing documents
            db_dir: Directory to store the vector database
        """
        try:
            self.docs_dir = docs_dir
            self.db_dir = db_dir
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            self.vector_db = None
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
        except Exception as e:
            logging.error(f"Error initializing DocumentProcessor: {str(e)}")
            raise
        
    def load_documents(self) -> List[Document]:
        """
        Load documents from the documents directory.
        
        Returns:
            List of loaded documents
        """
        try:
            # Configure loaders for different file types
            loaders = {
                ".txt": (TextLoader, {"encoding": "utf8"}),
                # Add more loaders for different file types as needed
                # ".pdf": (PyPDFLoader, {}),
                # ".docx": (Docx2txtLoader, {}),
            }
            
            # Create a directory loader with the appropriate loaders
            loader = DirectoryLoader(
                self.docs_dir,
                glob="**/*.*",
                loader_cls=loaders,
                show_progress=True,
                use_multithreading=True,
            )
            
            # Load the documents
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            logging.error(f"Error loading documents: {str(e)}")
            return []
    
    def process_documents(self) -> bool:
        """
        Process documents and store them in the vector database.
        
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Load documents
            documents = self.load_documents()
            
            if not documents:
                logging.warning("No documents to process")
                return False
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logging.info(f"Split into {len(chunks)} chunks")
            
            # Create or update the vector database
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
            
            # Persist the database
            self.vector_db.persist()
            logging.info(f"Vector database created with {len(chunks)} chunks")
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            return False
    
    def get_vector_db(self) -> Optional[Chroma]:
        """
        Get the vector database. If it doesn't exist, create it.
        
        Returns:
            Chroma vector database or None if it couldn't be created
        """
        # If the vector database is already loaded, return it
        if self.vector_db is not None:
            return self.vector_db
        
        # Check if the vector database exists
        if os.path.exists(self.db_dir):
            try:
                # Load the existing vector database
                self.vector_db = Chroma(
                    persist_directory=self.db_dir,
                    embedding_function=self.embeddings
                )
                logging.info("Loaded existing vector database")
                return self.vector_db
            except Exception as e:
                logging.error(f"Error loading vector database: {str(e)}")
        
        # If the vector database doesn't exist, create it
        if self.process_documents():
            return self.vector_db
        
        return None
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        vector_db = self.get_vector_db()
        
        if vector_db is None:
            logging.error("Vector database not available")
            return []
        
        try:
            # Retrieve documents
            retriever = vector_db.as_retriever(search_kwargs={"k": k})
            documents = retriever.get_relevant_documents(query)
            logging.info(f"Retrieved {len(documents)} documents for query: {query}")
            
            return documents
            
        except Exception as e:
            logging.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_document_content(self, documents: List[Document]) -> str:
        """
        Get the content of documents as a string.
        
        Args:
            documents: List of documents
            
        Returns:
            String containing the content of the documents
        """
        if not documents:
            return "No relevant documents found."
        
        content = "\n\n".join([doc.page_content for doc in documents])
        return content
