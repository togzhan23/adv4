import streamlit as st
import logging
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_openai import ChatOpenAI
import chromadb
import os
from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
from sentence_transformers import SentenceTransformer
from typing import List
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)

# Setup ChromaDB client and embedding function
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
llm_model = "llama3.2"

class ChromaDBEmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input)


embedding = ChromaDBEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  
)

# Define collection in ChromaDB
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)


def add_document_to_collection(documents, ids):
    try:
        embeddings = [embedding(doc) for doc in documents]
        collection.add(documents=documents, embeddings=[vec[0].tolist() for vec in embeddings], ids=ids)
    except Exception as e:
        logging.error(f"Error adding document: {e}")
        raise


def query_documents_from_chromadb(query_text, n_results=3):  # Fetch top 3 matches
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"], results["metadatas"]


def query_with_ollama(prompt, model_name):
    try:
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        logging.error(f"Error with Ollama query: {e}")
        return f"Error with Ollama API: {e}"


def retrieve_and_answer(query_text, model_name):
    try:
        retrieved_docs, metadatas = query_documents_from_chromadb(query_text)
        context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
        article_id = metadatas[0] if metadatas else "Unknown Article"
        augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
        response = query_with_ollama(augmented_prompt, model_name)
        return response, article_id
    except Exception as e:
        logging.error(f"Error in retrieve_and_answer: {e}")
        return f"Error: {e}", "Unknown Article"



#--------------------------------
# Define user validation functions
def validate_user(user_id: int, addresses: List) -> bool:
    """Validate user using historical addresses."""
    return True

def validate_user2(user_id: int, addresses: List) -> bool:
    """Validate user using historical addresses."""
    return False

# Create an instance of ChatOpenAI and bind the validation tools
llm = ChatOpenAI(
    api_key="ollama",
    model="llama3.2",
    base_url="http://localhost:11434/v1",
)

# Bind the validation tools to the LLM
llm = llm.bind_tools([validate_user, validate_user2])

#--------------------------------


def query_chromadb(query_texts, n_results=1):
    """Accepts multiple queries and retrieves corresponding documents."""
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

def query_ollama(prompt):
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

#Multi Query
def multi_query_rag_pipeline(query_text):
    """Handles multiple queries for RAG Fusion."""
    
    # Generate multiple query variations
    queries = [
        f"Extract key details related to '{query_text}' from the document.",
        f"Provide context and details about '{query_text}' in relation to the topic.",
        f"Retrieve relevant sections that talk about '{query_text}' from the document."
    ]
    
    # Retrieve documents for each query
    retrieved_docs, metadata = query_chromadb(queries, n_results=3)  # Retrieve top 3 docs for each query
    
    # RAG Fusion
    context = " ".join([doc for sublist in retrieved_docs for doc in sublist]) if retrieved_docs else "No relevant documents found."
    
    # Construct prompt for Ollama model
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    
    # Get response from Ollama
    response = query_ollama(augmented_prompt)
    return response


# Streamlit web interface
st.title("Chat with Ollama")

# Select a model
model = st.sidebar.selectbox("Choose a model", ["llama3.2", "phi3", "mistral"])

if not model:
    st.warning("Please select a model.")

# Add new document to ChromaDB menu
menu = st.sidebar.selectbox("Choose an action", ["Show Data", "Ask Questions", "Add Document to ChromaDB"])

if menu == "Add Document to ChromaDB":
    st.subheader("Add a New Document to ChromaDB")
    new_doc = st.text_area("Enter the new document:")
    if st.button("Add Document"):
        if new_doc:
            doc_id = f"doc{len(collection.get()['documents']) + 1}"  # New document ID based on current length
            add_document_to_collection([new_doc], [doc_id])
            st.success(f"Document added successfully with ID {doc_id}")
        else:
            st.warning("Please enter a document before adding.")

elif menu == "Ask Questions":
    query = st.text_input("Ask a question with context:")
    if query:
        # Use multi-query and RAG fusion pipeline
        response = multi_query_rag_pipeline(query)
        st.write("Response:", response)

elif menu == "Show Data":
    st.subheader("Stored Documents")
    documents = collection.get()["documents"]
    if documents:
        for i, doc in enumerate(documents, start=1):
            st.write(f"{i}. {doc}")
    else:
        st.write("No data available!")

# Handle chat interaction
def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

if 'messages' not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st.spinner("Thinking...")

        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
        

#--------------------------------
        # Call the LLM with validation
        result = llm.invoke(
            "Could you validate user 123? They previously lived at "
            "123 Fake St in Boston MA and 234 Pretend Boulevard in "
            "Houston TX."
        )

#--------------------------------
        
        response = stream_chat(model, messages)

        if response:
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
