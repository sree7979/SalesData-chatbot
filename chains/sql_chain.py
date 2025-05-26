"""
SQL generation chain for converting natural language to SQL queries.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, Tuple
import re
import os

from utils.db_utils import DatabaseManager
from utils.prompt_utils import PromptManager

class SQLGenerationChain:
    """
    A chain for generating SQL queries from natural language questions.
    """
    def __init__(self, db_manager: DatabaseManager, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the SQL generation chain.
        
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
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0)
        
        # Initialize LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.prompt_manager.get_sql_generation_prompt("{question}"))
        )
    
    def extract_sql_query(self, llm_response: str) -> str:
        """
        Extract the SQL query from the LLM response.
        
        Args:
            llm_response: Response from the LLM
            
        Returns:
            Extracted SQL query
        """
        # Extract SQL query from the response using regex
        sql_pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(sql_pattern, llm_response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no SQL code block is found, try to find any SQL-like content
        # This is a fallback in case the LLM doesn't format the response as expected
        lines = llm_response.split('\n')
        sql_lines = []
        capturing = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if this line looks like the start of a SQL query
            if not capturing and (line.upper().startswith("SELECT") or 
                                 line.upper().startswith("WITH")):
                capturing = True
                
            if capturing:
                sql_lines.append(line)
                
                # If we see a semicolon, it might be the end of the query
                if line.endswith(';'):
                    break
        
        if sql_lines:
            return '\n'.join(sql_lines)
            
        # If all else fails, return the original response
        return llm_response
    
    def validate_sql_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the SQL query to ensure it's safe to execute.
        
        Args:
            query: SQL query to validate
            
        Returns:
            Tuple containing:
            - Boolean indicating if the query is valid
            - Error message if the query is invalid, None otherwise
        """
        # Check if the query is empty
        if not query or query.strip() == "":
            return False, "Empty query"
        
        # Check if the query is a SELECT query (for safety)
        if not query.strip().upper().startswith("SELECT") and not query.strip().upper().startswith("WITH"):
            return False, "Only SELECT queries are allowed for safety reasons"
        
        # Check for dangerous keywords that might modify the database
        dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        for keyword in dangerous_keywords:
            if keyword in query.upper():
                return False, f"Dangerous keyword detected: {keyword}"
        
        return True, None
    
    def generate_sql(self, question: str) -> Dict[str, Any]:
        """
        Generate a SQL query from a natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary containing:
            - question: Original question
            - sql_query: Generated SQL query
            - is_valid: Boolean indicating if the query is valid
            - error: Error message if the query is invalid
        """
        # Generate SQL query using LLM
        response = self.chain.invoke({"question": question})
        llm_response = response["text"]
        
        # Extract SQL query from the response
        sql_query = self.extract_sql_query(llm_response)
        
        # Validate the SQL query
        is_valid, error = self.validate_sql_query(sql_query)
        
        return {
            "question": question,
            "sql_query": sql_query,
            "is_valid": is_valid,
            "error": error
        }
    
    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return the results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Dictionary containing:
            - results: Query results
            - error: Error message if the query failed
        """
        # Validate the SQL query
        is_valid, error = self.validate_sql_query(sql_query)
        
        if not is_valid:
            return {
                "results": [],
                "error": error
            }
        
        # Execute the query
        results, error = self.db_manager.execute_query(sql_query)
        
        return {
            "results": results,
            "error": error
        }
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question by generating and executing a SQL query.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary containing:
            - question: Original question
            - sql_query: Generated SQL query
            - results: Query results
            - error: Error message if the query failed
        """
        # Generate SQL query
        sql_result = self.generate_sql(question)
        
        if not sql_result["is_valid"]:
            return {
                "question": question,
                "sql_query": sql_result["sql_query"],
                "results": [],
                "error": sql_result["error"]
            }
        
        # Execute SQL query
        execution_result = self.execute_sql(sql_result["sql_query"])
        
        return {
            "question": question,
            "sql_query": sql_result["sql_query"],
            "results": execution_result["results"],
            "error": execution_result["error"]
        }
