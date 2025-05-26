"""
RAG chain for retrieving documents and generating responses.
"""
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END

from utils.document_utils import DocumentProcessor

# Define the state for the RAG graph
class RAGState(TypedDict):
    """State for the RAG graph."""
    question: str
    documents: List[Document]
    answer: str
    error: str

# RAG prompt template
RAG_PROMPT_TEMPLATE = """
You are an expert sales analyst who answers questions based on the provided documents.

User Question:
{question}

Relevant Documents:
{context}

Please provide a comprehensive answer to the user's question based on the information in the documents.
If the documents don't contain information to answer the question, say so clearly.
Use clear, concise language that a business user would understand.

Your answer:
"""

class RAGChain:
    """
    A chain for retrieving documents and generating responses using RAG.
    """
    def __init__(self, doc_processor: DocumentProcessor, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the RAG chain.
        
        Args:
            doc_processor: Document processor instance
            model_name: Name of the Gemini model to use
        """
        self.doc_processor = doc_processor
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        self.prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.graph = self._build_graph()
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve documents relevant to the question.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with retrieved documents
        """
        try:
            question = state["question"]
            documents = self.doc_processor.retrieve_documents(question, k=3)
            
            return {"documents": documents}
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            logging.error(error_msg)
            return {"documents": [], "error": error_msg}
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate an answer based on the retrieved documents.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with generated answer
        """
        try:
            question = state["question"]
            documents = state["documents"]
            
            if not documents:
                return {"answer": "I couldn't find any relevant information to answer your question."}
            
            # Get document content as a string
            context = self.doc_processor.get_document_content(documents)
            
            # Generate answer using LLM
            prompt_value = self.prompt.format(question=question, context=context)
            response = self.llm.invoke(prompt_value)
            
            return {"answer": response.content}
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            logging.error(error_msg)
            return {"answer": "I encountered an error while trying to answer your question.", "error": error_msg}
    
    def _should_end(self, state: RAGState) -> bool:
        """
        Determine if the graph should end.
        
        Args:
            state: Current state
            
        Returns:
            Boolean indicating if the graph should end
        """
        # End if there's an error or if an answer has been generated
        return bool(state.get("error")) or bool(state.get("answer"))
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for RAG.
        
        Returns:
            StateGraph instance
        """
        # Create a new graph
        graph = StateGraph(RAGState)
        
        # Add nodes
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        graph.add_edge("retrieve_documents", "generate_answer")
        graph.add_conditional_edges(
            "generate_answer",
            self._should_end,
            {
                True: END,
                False: "retrieve_documents"  # This should never happen with our current implementation
            }
        )
        
        # Set the entry point
        graph.set_entry_point("retrieve_documents")
        
        return graph.compile()
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question using the RAG chain.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing:
            - question: Original question
            - answer: Generated answer
            - error: Error message if an error occurred
        """
        # Initialize the state
        initial_state: RAGState = {
            "question": question,
            "documents": [],
            "answer": "",
            "error": ""
        }
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "question": question,
                "answer": result["answer"],
                "error": result.get("error", "")
            }
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logging.error(error_msg)
            
            return {
                "question": question,
                "answer": "I encountered an error while processing your question.",
                "error": error_msg
            }
