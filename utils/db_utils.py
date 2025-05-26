"""
Database utilities for interacting with the SQLite database.
"""
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

class DatabaseManager:
    """
    A class to manage database connections and operations.
    """
    def __init__(self, db_path: str):
        """
        Initialize the database manager with the path to the SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        
        Returns:
            A SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Execute an SQL query and return the results as a list of dictionaries.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple containing:
            - List of dictionaries with query results (each row as a dict)
            - Error message if an error occurred, None otherwise
        """
        try:
            conn = self.get_connection()
            # Execute the query and fetch results
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Convert results to a list of dictionaries
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return results, None
            
        except sqlite3.Error as e:
            # Return empty results and the error message
            return [], str(e)
    
    def execute_query_to_df(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute an SQL query and return the results as a pandas DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple containing:
            - DataFrame with query results (or None if error)
            - Error message if an error occurred, None otherwise
        """
        try:
            conn = self.get_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df, None
            
        except (sqlite3.Error, pd.errors.DatabaseError) as e:
            # Return None and the error message
            return None, str(e)
    
    def get_table_schema(self, table_name: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Get the schema for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Tuple containing:
            - List of dictionaries with column information
            - Error message if an error occurred, None otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            
            for row in cursor.fetchall():
                column_info = {
                    "name": row["name"],
                    "type": row["type"],
                    "primary_key": bool(row["pk"]),
                    "nullable": not bool(row["notnull"])
                }
                columns.append(column_info)
            
            conn.close()
            return columns, None
            
        except sqlite3.Error as e:
            return [], str(e)
    
    def get_all_tables(self) -> List[str]:
        """
        Get a list of all tables in the database.
        
        Returns:
            List of table names
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tables
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get foreign key information for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of dictionaries with foreign key information
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = []
        
        for row in cursor.fetchall():
            fk_info = {
                "column": row["from"],
                "referenced_table": row["table"],
                "referenced_column": row["to"]
            }
            foreign_keys.append(fk_info)
        
        conn.close()
        return foreign_keys
    
    def get_database_schema(self) -> Dict[str, Any]:
        """
        Get the complete database schema including all tables, columns, and relationships.
        
        Returns:
            Dictionary with complete database schema information
        """
        schema = {}
        tables = self.get_all_tables()
        
        for table in tables:
            columns, _ = self.get_table_schema(table)
            foreign_keys = self.get_foreign_keys(table)
            
            schema[table] = {
                "columns": columns,
                "foreign_keys": foreign_keys
            }
        
        return schema
    
    def get_schema_as_string(self) -> str:
        """
        Get a string representation of the database schema for use in prompts.
        
        Returns:
            String representation of the database schema
        """
        schema = self.get_database_schema()
        schema_str = "Database Schema:\n\n"
        
        for table_name, table_info in schema.items():
            schema_str += f"Table: {table_name}\n"
            schema_str += "Columns:\n"
            
            for column in table_info["columns"]:
                pk_str = " (Primary Key)" if column["primary_key"] else ""
                nullable_str = " (Nullable)" if column["nullable"] else ""
                schema_str += f"  - {column['name']} ({column['type']}){pk_str}{nullable_str}\n"
            
            if table_info["foreign_keys"]:
                schema_str += "Foreign Keys:\n"
                for fk in table_info["foreign_keys"]:
                    schema_str += f"  - {fk['column']} references {fk['referenced_table']}({fk['referenced_column']})\n"
            
            schema_str += "\n"
        
        return schema_str
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> str:
        """
        Get a string representation of sample data from a table.
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            
        Returns:
            String representation of sample data
        """
        df, error = self.execute_query_to_df(f"SELECT * FROM {table_name} LIMIT {limit}")
        
        if error:
            return f"Error retrieving sample data: {error}"
        
        return f"Sample data from {table_name}:\n{df.to_string()}\n"
