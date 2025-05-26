"""
Main Streamlit application for the Smart Sales Data Chatbot.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import logging
from dotenv import load_dotenv
import json

from utils.db_utils import DatabaseManager
from utils.document_utils import DocumentProcessor
from chains.sql_chain import SQLGenerationChain
from chains.summary_chain import ResultSummaryChain
from chains.rag_chain import RAGChain
from chains.router_chain import RouterChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Google Generative AI with API key
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Smart Sales Data Chatbot",
    page_icon="💬",
    layout="wide"
)

# Custom CSS to make the interface more lively
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #f5f7ff;
        color: #1e1e1e;
    }
    
    /* Header styling */
    h1 {
        color: #4a6bdf !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #e8eeff;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4a6bdf;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3a5bcf;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 20px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e1e9ff;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f0f4ff;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 20px;
        border: 2px solid #4a6bdf;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #e1e9ff;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid #e1e9ff;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        border: 1px solid #e1e9ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Hi there! I'm your Smart Sales Data Chatbot. I can answer questions about your sales data and documents. Try asking me something like 'What are the total sales for each product category?' or 'What was our revenue in 2023?'"}
    ]

if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager("sales_demo.db")

if "doc_processor" not in st.session_state:
    try:
        st.session_state.doc_processor = DocumentProcessor("docs", "chroma_db")
        # Process documents to create vector database
        with st.spinner("Processing documents for RAG..."):
            st.session_state.doc_processor.process_documents()
    except Exception as e:
        st.error(f"Error initializing DocumentProcessor: {str(e)}")
        st.stop()

if "sql_chain" not in st.session_state:
    st.session_state.sql_chain = SQLGenerationChain(st.session_state.db_manager)

if "summary_chain" not in st.session_state:
    st.session_state.summary_chain = ResultSummaryChain(st.session_state.db_manager)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RAGChain(st.session_state.doc_processor)

if "router_chain" not in st.session_state:
    st.session_state.router_chain = RouterChain()

# Function to process a question
def process_question(question):
    """Process a natural language question and update the chat history."""
    # Add user question to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Show a spinner while processing
    with st.spinner("Thinking..."):
        # Route the question to the appropriate system
        route_result = st.session_state.router_chain.route_question(question)
        route = route_result["route"]
        
        # Check if there was an error in routing
        if route_result["error"]:
            error_message = f"Error routing question: {route_result['error']}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            return
        
        # Process the question based on the routing decision
        if route == "sql":
            # Process with SQL chain
            sql_result = st.session_state.sql_chain.process_question(question)
            
            # Check if there was an error
            if sql_result["error"]:
                error_message = f"Error: {sql_result['error']}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                return
            
            # Generate summary
            summary_result = st.session_state.summary_chain.summarize_results(
                question=question,
                sql_query=sql_result["sql_query"],
                results=sql_result["results"]
            )
            
            # Create a response with SQL query, results, and summary
            response = {
                "type": "sql",
                "summary": summary_result["summary"],
                "sql_query": sql_result["sql_query"],
                "results": sql_result["results"]
            }
            
        elif route == "rag":
            # Process with RAG chain
            rag_result = st.session_state.rag_chain.process_question(question)
            
            # Check if there was an error
            if rag_result["error"]:
                error_message = f"Error: {rag_result['error']}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                return
            
            # Create a response with the RAG answer
            response = {
                "type": "rag",
                "summary": rag_result["answer"],
                "documents": "Retrieved from document knowledge base"
            }
            
        else:
            # Unknown route
            error_message = f"Could not determine how to process your question. Please try rephrasing it."
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            return
        
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# App title and description
st.title("💬 Smart Sales Data Chatbot ✨")
st.markdown("""
<div style="background-color: #e1e9ff; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <h3 style="margin-top: 0; color: #4a6bdf;">Welcome to your AI-powered sales assistant! 🚀</h3>
    <p>Ask questions about your sales data in natural language, and get powerful insights instantly:</p>
    <ul>
        <li>📊 <b>Analyze sales data</b> with simple questions</li>
        <li>📝 <b>Query documents</b> for strategic information</li>
        <li>📈 <b>Visualize results</b> with interactive charts</li>
        <li>💡 <b>Get insights</b> in plain English</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    # Database Information
    st.markdown("<h3 style='color: #4a6bdf;'>📊 Database Information</h3>", unsafe_allow_html=True)
    
    # Display database schema
    with st.expander("Database Schema", expanded=False):
        st.code(st.session_state.db_manager.get_schema_as_string())
    
    # Display sample data
    with st.expander("Sample Data", expanded=False):
        st.subheader("Products")
        products_df, _ = st.session_state.db_manager.execute_query_to_df("SELECT * FROM products LIMIT 5")
        st.dataframe(products_df)
        
        st.subheader("Customers")
        customers_df, _ = st.session_state.db_manager.execute_query_to_df("SELECT * FROM customers LIMIT 5")
        st.dataframe(customers_df)
        
        st.subheader("Sales")
        sales_df, _ = st.session_state.db_manager.execute_query_to_df("SELECT * FROM sales LIMIT 5")
        st.dataframe(sales_df)
    
    # Document Information
    st.markdown("<h3 style='color: #4a6bdf;'>📝 Document Information</h3>", unsafe_allow_html=True)
    
    # Display available documents
    with st.expander("Available Documents", expanded=False):
        st.markdown("""
        - **Sales Report 2023**: Annual sales report with performance by category and region
        - **Product Strategy 2024**: Strategic plan for product development and marketing
        """)
    
    # Example SQL Questions
    st.markdown("<h3 style='color: #4a6bdf;'>🔍 Example SQL Questions</h3>", unsafe_allow_html=True)
    sql_questions = [
        "What are the total sales for each product category?",
        "Who are the top 5 customers by total purchase amount?",
        "What were the total sales in the East region in 2023?",
        "How many sales were made for each product category by region?",
        "What's the average order amount by month in 2023?"
    ]
    
    for question in sql_questions:
        if st.button(question, key=f"sql_{question}"):
            # Add the question to the chat input
            st.session_state.question = question
            # Use the on_submit function to process the question
            process_question(question)
    
    # Example RAG Questions
    st.markdown("<h3 style='color: #4a6bdf;'>📚 Example Document Questions</h3>", unsafe_allow_html=True)
    rag_questions = [
        "What was our revenue in 2023?",
        "Which product category had the highest growth rate?",
        "What are the key strategic initiatives for 2024?",
        "Who were our top 5 customers in 2023?",
        "What is the plan for the Clothing category in 2024?"
    ]
    
    for question in rag_questions:
        if st.button(question, key=f"rag_{question}"):
            # Add the question to the chat input
            st.session_state.question = question
            # Use the on_submit function to process the question
            process_question(question)

# Chat input
st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
question = st.chat_input("Ask a question about your sales data or documents... 💬")
if question:
    st.session_state.question = question
    process_question(question)

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:  # assistant
        with st.chat_message("assistant"):
            if isinstance(message["content"], str):
                # Display simple text response (usually an error)
                st.write(message["content"])
            else:
                # Display structured response
                response = message["content"]
                
                # Display the summary
                st.write(response["summary"])
                
                # Handle different response types
                if response.get("type") == "sql":
                    # Display SQL-specific information
                    with st.expander("View SQL Query"):
                        st.code(response["sql_query"], language="sql")
                    
                    with st.expander("View Raw Results"):
                        # Convert results to DataFrame for display
                        if response["results"]:
                            results_df = pd.DataFrame(response["results"])
                            st.dataframe(results_df)
                            
                            # Try to create a visualization based on the data
                            try:
                                # Only create visualizations for certain types of data
                                if len(results_df) > 0 and len(results_df) <= 20:
                                    cols = results_df.columns.tolist()
                                    
                                    # Check if we have numeric and categorical columns for a chart
                                    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
                                    categorical_cols = [c for c in cols if c not in numeric_cols]
                                    
                                    if numeric_cols and categorical_cols:
                                        # Create a bar chart
                                        st.subheader("Visualization")
                                        fig = px.bar(
                                            results_df, 
                                            x=categorical_cols[0], 
                                            y=numeric_cols[0],
                                            title=f"{categorical_cols[0]} vs {numeric_cols[0]}"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.write(f"Could not create visualization: {str(e)}")
                        else:
                            st.write("No results to display.")
                
                elif response.get("type") == "rag":
                    # Display RAG-specific information
                    with st.expander("Source"):
                        st.write(response.get("documents", "Information retrieved from document knowledge base"))
                
                else:
                    # Handle legacy or unknown response types
                    if "sql_query" in response:
                        with st.expander("View SQL Query"):
                            st.code(response["sql_query"], language="sql")
                    
                    if "results" in response:
                        with st.expander("View Raw Results"):
                            if response["results"]:
                                results_df = pd.DataFrame(response["results"])
                                st.dataframe(results_df)
                            else:
                                st.write("No results to display.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; margin-top: 2rem;">
    <p style="color: #4a6bdf; font-weight: bold;">✨ Smart Sales Data Chatbot ✨</p>
    <p>Powered by Google Gemini and LangChain</p>
    <p style="font-size: 0.8rem; color: #666;">© 2025 | Made with ❤️ for better sales insights</p>
</div>
""", unsafe_allow_html=True)
