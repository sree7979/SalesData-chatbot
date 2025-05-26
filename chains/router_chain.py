"""
Router chain for determining whether to use SQL or RAG.
"""
from typing import Dict, Any, TypedDict, Literal
import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# Define the state for the router graph
class RouterState(TypedDict):
    """State for the router graph."""
    question: str
    route: Literal["sql", "rag", "unknown"]
    error: str

# Router prompt template
ROUTER_PROMPT_TEMPLATE = """
You are an expert system that determines whether a user's question should be answered using SQL queries on a database or by retrieving information from documents.

The system has two capabilities:
1. SQL: For questions about sales data, metrics, statistics, and quantitative analysis that can be answered by querying a database.
2. RAG (Retrieval Augmented Generation): For questions about company reports, strategies, policies, and qualitative information that would be found in documents.

Database Schema:
- products (product_id, product_name, category)
- customers (customer_id, customer_name, region)
- sales (order_id, product_id, customer_id, amount, order_date)

Available Documents:
- Sales reports
- Product strategies
- Marketing plans
- Company policies

User Question:
{question}

Determine whether this question should be routed to the SQL system or the RAG system.
Respond with exactly one word: either "sql" or "rag".
"""

class RouterChain:
    """
    A chain for routing questions to the appropriate system (SQL or RAG).
    """
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the router chain.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.prompt = PromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
        self.graph = self._build_graph()
    
    def _route_question(self, state: RouterState) -> RouterState:
        """
        Route the question to the appropriate system.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with routing decision
        """
        try:
            question = state["question"]
            
            # Generate routing decision using LLM
            prompt_value = self.prompt.format(question=question)
            response = self.llm.invoke(prompt_value)
            
            # Extract the routing decision
            route = response.content.strip().lower()
            
            # Validate the routing decision
            if route not in ["sql", "rag"]:
                logging.warning(f"Invalid routing decision: {route}. Defaulting to 'unknown'.")
                return {"route": "unknown"}
            
            return {"route": route}
            
        except Exception as e:
            error_msg = f"Error routing question: {str(e)}"
            logging.error(error_msg)
            return {"route": "unknown", "error": error_msg}
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for routing.
        
        Returns:
            StateGraph instance
        """
        # Create a new graph
        graph = StateGraph(RouterState)
        
        # Add nodes
        graph.add_node("route_question", self._route_question)
        
        # Add edges
        graph.add_edge("route_question", END)
        
        # Set the entry point
        graph.set_entry_point("route_question")
        
        return graph.compile()
    
    def route_question(self, question: str) -> Dict[str, Any]:
        """
        Route a question to the appropriate system.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing:
            - question: Original question
            - route: Routing decision (sql, rag, or unknown)
            - error: Error message if an error occurred
        """
        # Check if the question is a greeting
        greetings = ["hi", "hello", "hey", "greetings", "howdy"]
        if question.lower() in greetings or question.lower().startswith(tuple(g + " " for g in greetings)):
            return {
                "question": question,
                "route": "unknown",
                "error": "I'm a sales data chatbot. You can ask me questions about your sales data or documents. For example, try asking 'What are the total sales for each product category?' or 'What was our revenue in 2023?'"
            }
        
        # Initialize the state
        initial_state: RouterState = {
            "question": question,
            "route": "unknown",
            "error": ""
        }
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "question": question,
                "route": result["route"],
                "error": result.get("error", "")
            }
            
        except Exception as e:
            error_msg = f"Error routing question: {str(e)}"
            logging.error(error_msg)
            
            return {
                "question": question,
                "route": "unknown",
                "error": error_msg
            }
