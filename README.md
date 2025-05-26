# Smart Sales Data Chatbot
![smart-bot](https://github.com/user-attachments/assets/cd97618f-6ef0-4ed9-8d83-fa097d6c2409)


## Features

- **Natural Language Understanding**: Ask questions about your sales data and documents in plain English
- **Intelligent Routing**: Automatically determines whether to use SQL or document retrieval
- **SQL Generation**: Converts data questions to SQL queries for database analysis
- **Retrieval Augmented Generation (RAG)**: Finds relevant information in documents for knowledge questions
- **Data Visualization**: Displays SQL results with appropriate visualizations
- **Comprehensive Summaries**: Provides insights and analysis of query results and document information
- **User-Friendly Interface**: Simple chat interface with example questions for both data and documents

## Architecture

The application is built with:

- **Python**: Core programming language
- **LangChain & LangGraph**: Framework for LLM interactions
- **Google Gemini**: LLM for natural language understanding and generation
- **OpenAI Embeddings**: For document vector embeddings
- **SQLite**: Database engine
- **Streamlit**: Web interface
- **Pandas & Plotly**: Data manipulation and visualization

## Project Structure

```
smart-summary-bot/
├── app.py                 # Main Streamlit application
├── sales_demo.db          # SQLite database with sales data
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables file
├── docs/                  # Document knowledge base
│   ├── sales_report_2023.txt     # Annual sales report
│   └── product_strategy_2024.txt # Product strategy document
├── utils/
│   ├── __init__.py
│   ├── db_utils.py        # Database interaction functions
│   ├── document_utils.py  # Document processing for RAG
│   └── prompt_utils.py    # Prompt templates and helpers
├── chains/
│   ├── __init__.py
│   ├── sql_chain.py       # SQL generation chain
│   ├── summary_chain.py   # Result summarization chain
│   ├── rag_chain.py       # RAG implementation with LangGraph
│   └── router_chain.py    # Question routing logic
├── chroma_db/             # Vector database for document embeddings
└── README.md              # Project documentation
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file with your Google API key and OpenAI API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     OPENAI_API_KEY=your_openai_api_key_here
     ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Start the application with `streamlit run app.py`
2. Type your question in the chat input or select an example question
3. View the generated SQL, results, and natural language summary
4. Explore visualizations of the data when available

## Example Questions

### SQL Questions (Database Analysis)

- "What are the total sales for each product category?"
- "Who are the top 5 customers by total purchase amount?"
- "What were the total sales in the East region in 2023?"
- "How many sales were made for each product category by region?"
- "What's the average order amount by month in 2023?"

### Document Questions (Knowledge Base)

- "What was our revenue in 2023?"
- "Which product category had the highest growth rate?"
- "What are the key strategic initiatives for 2024?"
- "Who were our top 5 customers in 2023?"
- "What is the plan for the Clothing category in 2024?"

## Data Sources

### Database Schema

The SQLite database (`sales_demo.db`) contains:

- **Products table**: Products with IDs, names, and categories
- **Customers table**: Customers with IDs, names, and regions
- **Sales table**: Sales records linking products to customers with amounts and dates

### Document Knowledge Base

The document knowledge base (`docs/` directory) contains:

- **Sales Report 2023**: Annual sales report with performance by category and region
- **Product Strategy 2024**: Strategic plan for product development and marketing

You can add more documents to the `docs/` directory to expand the knowledge base. The system will automatically process and index them for retrieval.

## Technical Details

### LLM Models

- **Google Gemini 1.5 Flash**: Used for natural language understanding, SQL generation, and summarization
- **OpenAI Embeddings**: Used for document vector embeddings in the RAG system

### Chains

- **Router Chain**: Determines whether to use SQL or RAG based on the question
- **SQL Chain**: Generates SQL queries from natural language questions
- **RAG Chain**: Retrieves relevant documents and generates answers
- **Summary Chain**: Summarizes SQL query results in natural language

## Troubleshooting

If you encounter any issues:

1. Verify that your API keys are correctly set in the `.env` file
2. Check that all dependencies are installed
3. Ensure that the database file `sales_demo.db` is in the correct location
4. Verify that the document files are in the `docs/` directory


