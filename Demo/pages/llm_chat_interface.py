import streamlit as st
import neo4j
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem
import time

# set up OpenAI API key
import dotenv
import os
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="LLM Chat Interface",
    page_icon="🤖",
    layout="wide"
)

# Neo4j connection configuration
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_DATABASE = st.secrets["NEO4J_DATABASE"]

URI = NEO4J_URI
AUTH = (NEO4J_USER, NEO4J_PASSWORD)
INDEX_NAME = "entity_embeddings"

# Custom CSS
st.markdown("""
<style>
    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #4f46e5 0%, #6d28d9 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 600;
    }

       /* Chat message container */
    .chat-message {
        max-width: 80%;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 1rem;
        line-height: 1.4;
        background: #f9f9f9; /* light neutral gray for all messages */
        color: #111; /* dark text */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* User message (right aligned, but no color difference) */
    .user-message {
        background: #f9f9f9;
        margin-left: auto;
        border-radius: 8px 8px 0px 8px;
    }

    /* Assistant message (left aligned, but no color difference) */
    .assistant-message {
        background: #f9f9f9;
        margin-right: auto;
        border-radius: 8px 8px 8px 0px;
    }

    /* System message */
    .system-message {
        background: #f0f0f0;
        margin-right: auto;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #333;
        border-left: 3px solid #ccc;
    }

    /* Retrieval info block */
    .retrieval-info {
        background: #f7f7f7;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #333;
    }

    /* Model status badge */
    .model-status {
        background: #f0f0f0;
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: 500;
    }

    /* Error badge */
    .error-status {
        background: #f0f0f0;
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 LLM Chat Interface</h1>
    <h3>Chat with your Biomedical Knowledge Graph using Advanced GraphRAG</h3>
</div>
""", unsafe_allow_html=True)

def result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    """Format the extracted Neo4j records into more structured 
    RetrieverResultItem for GraphRAG"""
    
    # Extract data from the record
    name = record.get('name', '')
    relationship = record.get('relationship', '')
    description = record.get('description', '')
    related_entity = record.get('related_entity', '')
    other_relationship = record.get('other_relationship', '')
    
    # Create a structured content string
    content_parts = []
    if name:
        content_parts.append(f"Entity: {name}")
    if relationship and description:
        content_parts.append(f"Relationship: {relationship}")
        content_parts.append(f"Description: {description}")
    if related_entity and other_relationship:
        content_parts.append(f"Related: {related_entity} via {other_relationship}")
    
    content = "; ".join(content_parts) if content_parts else str(record)
    
    return RetrieverResultItem(
        content=content,
        metadata={
            "entity_name": name,
            "relationship_type": relationship,
            "description": description,
            "related_entity": related_entity,
            "score": record.get("score", 0.0),
        }
    )

@st.cache_resource
def initialize_graphrag():
    """Initialize GraphRAG system with caching"""
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            
            # Initialize embedder
            embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
            
            # Define retrieval query
            retrieval_query = """
            MATCH (node)-[r]->(related:Entity)
            OPTIONAL MATCH (related)-[r2]->(other:Entity)
            RETURN related.name as name, type(r) as relationship, r.description as description,
                   other.name as related_entity, type(r2) as other_relationship
            """
            
            # Create retriever
            retriever = VectorCypherRetriever(
                driver, 
                index_name=INDEX_NAME, 
                embedder=embedder, 
                retrieval_query=retrieval_query, 
                result_formatter=result_formatter
            )
            
            # Initialize LLM
            llm = OpenAILLM(model_name="gpt-4.1-mini", model_params={"temperature": 0}, api_key=os.getenv("OPENAI_API_KEY"))
            
            # Create GraphRAG
            rag = GraphRAG(
                retriever=retriever, 
                llm=llm
            )
            
            return rag, True, None
            
    except Exception as e:
        return None, False, str(e)

def display_chat_message(role, content, metadata=None):
    """Display a chat message with proper styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>LLM Response:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        st.markdown(f"""
        <div class="chat-message system-message">
            <strong>⚙️ System:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    

def main():
    # Sidebar for configuration
    st.sidebar.title("🔧 GraphRAG Configuration")
    
    # Model status check
    st.sidebar.subheader("System Status")
    
    # Check Neo4j connection
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
        st.sidebar.markdown('<span class="model-status">✅ Neo4j Connected</span>', unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.markdown('<span class="error-status">❌ Neo4j Connection Failed</span>', unsafe_allow_html=True)
        st.error(f"Cannot connect to Neo4j: {str(e)}")
        return
    
    # Chat configuration
    st.sidebar.subheader("Chat Settings")
    max_tokens = st.sidebar.slider("Max Response Length", 100, 1000, 500)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.rag_initialized = False
    
    # Initialize GraphRAG once
    if not st.session_state.rag_initialized:
        with st.spinner("⚙️ Initializing GraphRAG system..."):
            rag, success, error = initialize_graphrag()
            
            if success:
                st.session_state.rag = rag
                st.session_state.rag_initialized = True
                st.success("✅ GraphRAG system initialized successfully!")
            else:
                st.error(f"❌ Failed to initialize GraphRAG: {error}")
                st.info("Please ensure OpenAI is running and the required models are installed.")
                return
    

    st.subheader("💬 Chat with your Knowledge Graph")

    # --- Display existing chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):  # user or assistant
            st.markdown(msg["content"])
            
    # --- Input box at the bottom ---
    if prompt := st.chat_input("Ask a question about your biomedical knowledge graph..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.rag.search(query_text=prompt)
                    answer = response.answer
                                        
                    # Show answer
                    st.markdown(answer)
                    
                    # Save assistant message to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear chat button at the bottom
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
if __name__ == "__main__":
    main()