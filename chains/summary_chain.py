"""
Summary chain for converting SQL results to natural language summaries.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
import pandas as pd
import json
import os

from utils.db_utils import DatabaseManager
from utils.prompt_utils import PromptManager

class ResultSummaryChain:
    """
    A chain for summarizing SQL query results in natural language.
    """
    def __init__(self, db_manager: DatabaseManager, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the result summary chain.
        
        Args:
            db_manager: Database manager instance
            model_name: Name of the Gemini model to use
        """
        self.db_manager = db_manager
        self.model_name = model_name
        
        # Get database schema
        self.db_schema = db_manager.get_schema_as_string()
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(self.db_schema)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.2)
    
    def format_results_as_string(self, results: List[Dict[str, Any]]) -> str:
        """
        Format query results as a string for inclusion in the prompt.
        
        Args:
            results: Query results as a list of dictionaries
            
        Returns:
            Formatted string representation of the results
        """
        if not results:
            return "No results found."
        
        # Convert results to a pandas DataFrame for easier formatting
        try:
            df = pd.DataFrame(results)
            # Format as a string with both tabular and JSON representations
            tabular = df.to_string(index=False)
            json_str = json.dumps(results, indent=2)
            
            return f"Tabular format:\n{tabular}\n\nJSON format:\n{json_str}"
        except Exception as e:
            return f"Error formatting results: {str(e)}"
    
    def summarize_results(self, question: str, sql_query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize SQL query results in natural language.
        
        Args:
            question: Original natural language question
            sql_query: SQL query that was executed
            results: Query results as a list of dictionaries
            
        Returns:
            Dictionary containing:
            - question: Original question
            - sql_query: SQL query that was executed
            - summary: Natural language summary of the results
        """
        # Format results as a string
        results_str = self.format_results_as_string(results)
        
        # Get the summarization prompt
        prompt_text = self.prompt_manager.get_result_summarization_prompt(
            question=question,
            sql_query=sql_query,
            results=results_str
        )
        
        # Use the Google Generative AI API directly to bypass input validation
        try:
            # Generate the summary using the LLM directly
            response = self.llm.invoke(prompt_text)
            summary = response.content
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            summary = f"Error generating summary: {str(e)}"
        
        return {
            "question": question,
            "sql_query": sql_query,
            "summary": summary
        }
    
    def generate_visualizations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate visualization recommendations based on the query results.
        
        Args:
            results: Query results as a list of dictionaries
            
        Returns:
            Dictionary containing visualization recommendations
        """
        # This is a placeholder for future implementation
        # In a real implementation, this would analyze the data and suggest appropriate visualizations
        return {
            "recommended_viz_type": "bar_chart" if len(results) > 0 and len(results) <= 10 else "table",
            "viz_columns": list(results[0].keys()) if results else []
        }
