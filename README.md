# adv4
# Project: Multi-Query RAG Pipeline with Streamlit, ChromaDB, and Ollama
Assignment 4 Advanced Programming. Teamwork by Togzhan Oral and Yelnura Akhmetova.   

## Overview

This project integrates **Streamlit**, **ChromaDB**, **OllamaLLM**, and **LangChain** to create a powerful document retrieval system with multi-query and RAG fusion. The application enables users to add documents, query them with context, and receive responses using a multi-query RAG pipeline that aggregates the context from multiple document matches for more comprehensive answers.

### Key Features:
- **Multi-Query Handling**: Generate multiple query variations to retrieve the most relevant documents.
- **RAG Fusion**: Fuse the context from multiple documents to build a comprehensive response.
- **Document Management**: Upload and store documents in **ChromaDB** for easy retrieval.
- **Interactive Q&A**: Ask questions and receive answers through **Ollama** using the aggregated context.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/multi-query-rag.git
    ```

2. Navigate to the project directory:
    ```bash
    cd multi-query-rag
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure the **Ollama** API and **ChromaDB** are properly set up.

## Usage

1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Use the web interface to interact with the app:
    - **Show Data**: View stored documents in **ChromaDB**.
    - **Ask Questions**: Input a query to retrieve documents and get answers via **Ollama** using the multi-query and RAG fusion pipeline.
    - **Add Document to ChromaDB**: Upload a new document to be indexed and stored in **ChromaDB**.

## Features

### Multi-Query RAG Pipeline
- The app generates multiple variations of the input query and retrieves the top documents for each variation.
- It then fuses the context from these documents to provide a more comprehensive answer.

### Add Document to ChromaDB
- Upload documents via a text input field. The documents are processed and stored in **ChromaDB**.
- Documents are embedded using **SentenceTransformer** for efficient retrieval.

### Query and Answer
- Ask a question in the input field, and the system retrieves documents based on the multi-query approach.
- The retrieved document context is passed to **Ollama** to generate a response.

## Code Highlights

### Multi-Query Handling
- Generates multiple variations of the query to retrieve relevant documents for a more accurate answer.

### RAG Fusion
- Combines context from multiple documents into a single cohesive prompt for **Ollama**, ensuring better and more accurate answers.

### Document Management
- Supports both manual text input and file uploads for document management.
- Uses **ChromaDB** to store document embeddings generated using **SentenceTransformer**.

### Querying Documents
- Retrieves documents from **ChromaDB** based on similarity to the query text.
- Uses a multi-query approach to enhance the retrieval process.

### Streamlit Integration
- Provides an interactive web interface for querying, document management, and displaying results.
  
## Example Workflow

1. **Add a New Document**
    - Input a document into the provided text area or upload a file.
    - The document is processed and added to **ChromaDB**.

2. **Ask a Question**
    - Type a question in the input field.
    - The app retrieves relevant documents and uses **Ollama** to generate a response based on the aggregated context.

## Screenshot of the Interface

<img width="1440" alt="image" src="https://github.com/user-attachments/assets/b3f316af-5e76-48a8-abb2-693180882e63" />


## Conclusion

This project showcases the combination of **multi-query handling** and **RAG fusion** to answer complex questions based on multiple document contexts. Leveraging **ChromaDB**, **Ollama**, and **Streamlit**, this application provides an easy-to-use interface for document management, querying, and answering in a robust and scalable manner.
