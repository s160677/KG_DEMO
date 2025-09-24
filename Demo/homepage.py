import streamlit as st
import subprocess
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Biomedical Knowledge Graph Platform - DEMO",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .tech-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .launch-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 2rem;
        border-radius: 25px;
        border: none;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        margin: 1rem 0;
    }
    .coming-soon {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        font-size: 0.9rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 Biomedical Knowledge Graph Platform - DEMO</h1>
    <h3>Explore Complex Biomedical Relationships with AI-Powered Knowledge Graphs </h3>
</div>
""", unsafe_allow_html=True)

# Features Section with clickable feature cards
st.markdown("## Potential Platform Features")

# Function to show coming soon message
def show_coming_soon(feature_name):
    st.info(f"{feature_name} - Coming Soon!")
    st.markdown("""
    <div class="coming-soon">
        This feature is currently under development and will be available in a future update.
    </div>
    """, unsafe_allow_html=True)

# Create 2x2 grid for features
col1, col2 = st.columns(2)

with col1:
    # Feature 1: Create Knowledge Graph (COMING SOON)
    with st.container():
        if st.button("🔧 Create Knowledge Graph", use_container_width=True, key="create_kg"):
            show_coming_soon("Create Knowledge Graph")
        st.markdown("""
        <div class="feature-card">
            <p>Build and customize your own biomedical knowledge graphs by uploading your own research papers, clinical data, and scientific literature using advanced NLP and entity extraction techniques.</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature 3: ML Analytics (WORKING)
    with st.container():
        if st.button("📊 ML Analytics", use_container_width=True, key="ml_analytics"):
            st.switch_page("pages/ml_analysis.py")
        st.markdown("""
        <div class="feature-card">
            <p>Advanced machine learning analytics for knowledge graph insights. Perform link prediction, node classification, and graph neural network analysis on your biomedical data.</p>
        </div>
        """, unsafe_allow_html=True)

with col2:

    # Feature 2: Interactive Graph Visualization (WORKING)
    with st.container():
        if st.button("🎯 Interactive Graph Visualization", use_container_width=True, key="interactive_viz"):
            st.switch_page("pages/graph_visualization.py")
        st.markdown("""
        <div class="feature-card">
            <p>Explore complex biomedical relationships through dynamic, interactive graph visualizations. Zoom, filter, and navigate through thousands of entities and relationships with ease.</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature 4: LLM Interface (COMING SOON)
    with st.container():
        if st.button("🤖 LLM Chat Interface", use_container_width=True, key="llm_interface"):
            st.switch_page("pages/llm_chat_interface.py")
        st.markdown("""
        <div class="feature-card">
            <p>Chat with your knowledge graph using natural language. Ask questions and get intelligent responses from the LLM based on your knowledge graph data.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Biomedical Knowledge Graph Platform</strong> | Powered by Neo4j, AI, and Streamlit</p>
    <p>Advancing biomedical research through intelligent knowledge graphs</p>
</div>
""", unsafe_allow_html=True)