"""
Prompt utilities for the LLM interactions.
"""
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any

# SQL generation prompt template
SQL_GENERATION_TEMPLATE = """
You are an expert SQL query generator that helps convert natural language questions about sales data into SQL queries.

{schema}

The database contains information about products, customers, and sales. You need to generate a valid SQL query based on the user's question.

Now, generate a SQL query for the following question:
Question: {question}

SQL:
```sql
"""

# Result summarization prompt template
RESULT_SUMMARIZATION_TEMPLATE = """
You are an expert data analyst who explains SQL query results in clear, natural language.

The user asked the following question:
{question}

The SQL query that was executed:
```sql
{sql_query}
```

The query returned the following results:
{results}

Please provide a comprehensive summary of these results that answers the user's question.
Focus on the key insights, trends, and notable data points.
Use clear, concise language that a business user would understand.
If appropriate, suggest follow-up questions or additional analyses that might provide further insights.

Your summary:
"""

class PromptManager:
    """
    A class to manage prompt templates for the LLM.
    """
    def __init__(self, db_schema: str):
        """
        Initialize the prompt manager with the database schema.
        
        Args:
            db_schema: String representation of the database schema
        """
        self.db_schema = db_schema
        self.sql_prompt = PromptTemplate.from_template(SQL_GENERATION_TEMPLATE)
        self.summary_prompt = PromptTemplate.from_template(RESULT_SUMMARIZATION_TEMPLATE)
    
    def get_sql_generation_prompt(self, question: str) -> str:
        """
        Get the formatted SQL generation prompt.
        
        Args:
            question: User's natural language question
            
        Returns:
            Formatted prompt string
        """
        return self.sql_prompt.format(schema=self.db_schema, question=question)
    
    def get_result_summarization_prompt(self, question: str, sql_query: str, results: str) -> str:
        """
        Get the formatted result summarization prompt.
        
        Args:
            question: User's natural language question
            sql_query: SQL query that was executed
            results: String representation of the query results
            
        Returns:
            Formatted prompt string
        """
        return self.summary_prompt.format(
            question=question,
            sql_query=sql_query,
            results=results
        )
