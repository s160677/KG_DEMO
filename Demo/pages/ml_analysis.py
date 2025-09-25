import streamlit as st
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import hashlib

# Page configuration
st.set_page_config(
    page_title="ML Analytics - Link Prediction",
    page_icon="📊",
    layout="wide"
)

# Neo4j connection details
NEO4J_URI = "neo4j+s://29c0fe72.databases.neo4j.io"
NEO4J_USER = "29c0fe72"
NEO4J_PASSWORD = "D70-cnMF4HDGbhSIHBVdw9KOhVWlGmPauBN7EoEF-Z4"
NEO4J_DATABASE = "29c0fe72"   # change if you use another db

URI = NEO4J_URI
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# Initialize session state for persistent results
if 'ml_analysis_results' not in st.session_state:
    st.session_state.ml_analysis_results = None
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'model_cache_key' not in st.session_state:
    st.session_state.model_cache_key = None

# Custom CSS
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
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .success-metric {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .results-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>ML Analytics - Link Prediction</h1>
    <h3>Advanced Machine Learning Analysis for Biomedical Knowledge Graphs</h3>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_graph_features():
    """Extract features from the Neo4j graph for link prediction"""
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        
        # Get all existing relationships (positive examples)
        positive_edges = []
        result = driver.execute_query("""
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN a.name as source, b.name as target, type(r) as rel_type
        """, database="neo4j")
        
        for record in result.records:
            positive_edges.append({
                'source': record['source'],
                'target': record['target'],
                'rel_type': record['rel_type'],
                'exists': 1
            })
        
        # Get all nodes
        nodes_result = driver.execute_query("""
            MATCH (n:Entity)
            RETURN n.name as name, n.embedding as embedding
        """, database="neo4j")
        
        nodes = {}
        for record in nodes_result.records:
            nodes[record['name']] = {
                'embedding': record['embedding'] if record['embedding'] else None
            }
    
    # Generate negative examples
    node_names = list(nodes.keys())
    existing_pairs = set((edge['source'], edge['target']) for edge in positive_edges)
    
    negative_edges = []
    while len(negative_edges) < len(positive_edges):
        source = random.choice(node_names)
        target = random.choice(node_names)
        if source != target and (source, target) not in existing_pairs:
            negative_edges.append({
                'source': source,
                'target': target,
                'rel_type': 'negative',
                'exists': 0
            })
    
    return positive_edges, negative_edges, nodes

def calculate_node_features(nodes, edges):
    """Calculate node-level features for link prediction"""
    
    # Calculate degree centrality for each node
    degree_centrality = {}
    for node in nodes.keys():
        degree_centrality[node] = 0
    
    for edge in edges:
        if edge['exists'] == 1:
            degree_centrality[edge['source']] += 1
            degree_centrality[edge['target']] += 1
    
    # Calculate clustering coefficient (simplified)
    clustering_coeff = {}
    for node in nodes.keys():
        # Find neighbors
        neighbors = set()
        for edge in edges:
            if edge['exists'] == 1:
                if edge['source'] == node:
                    neighbors.add(edge['target'])
                elif edge['target'] == node:
                    neighbors.add(edge['source'])
        
        # Calculate clustering coefficient
        if len(neighbors) >= 2:
            neighbor_connections = 0
            for n1, n2 in combinations(neighbors, 2):
                if any((e['source'] == n1 and e['target'] == n2) or 
                       (e['source'] == n2 and e['target'] == n1) 
                       for e in edges if e['exists'] == 1):
                    neighbor_connections += 1
            max_connections = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeff[node] = neighbor_connections / max_connections if max_connections > 0 else 0
        else:
            clustering_coeff[node] = 0
    
    return degree_centrality, clustering_coeff

def create_link_prediction_features(positive_edges, negative_edges, nodes):
    """Create features for link prediction"""
    
    all_edges = positive_edges + negative_edges
    
    # Calculate node features
    degree_centrality, clustering_coeff = calculate_node_features(nodes, all_edges)
    
    features = []
    labels = []
    
    for edge in all_edges:
        source = edge['source']
        target = edge['target']
        
        # Node-level features
        source_degree = degree_centrality[source]
        target_degree = degree_centrality[target]
        source_clustering = clustering_coeff[source]
        target_clustering = clustering_coeff[target]
        
        # Embedding similarity (if available)
        embedding_similarity = 0
        if (nodes[source]['embedding'] is not None and 
            nodes[target]['embedding'] is not None):
            try:
                source_emb = np.array(nodes[source]['embedding'])
                target_emb = np.array(nodes[target]['embedding'])
                embedding_similarity = np.dot(source_emb, target_emb) / (
                    np.linalg.norm(source_emb) * np.linalg.norm(target_emb)
                )
            except:
                embedding_similarity = 0
        
        # Common neighbors
        source_neighbors = set()
        target_neighbors = set()
        
        for e in all_edges:
            if e['exists'] == 1:
                if e['source'] == source:
                    source_neighbors.add(e['target'])
                elif e['target'] == source:
                    source_neighbors.add(e['source'])
                if e['source'] == target:
                    target_neighbors.add(e['target'])
                elif e['target'] == target:
                    target_neighbors.add(e['source'])
        
        common_neighbors = len(source_neighbors.intersection(target_neighbors))
        
        # Jaccard similarity
        union_neighbors = source_neighbors.union(target_neighbors)
        jaccard_similarity = common_neighbors / len(union_neighbors) if len(union_neighbors) > 0 else 0
        
        # Preferential attachment
        preferential_attachment = len(source_neighbors) * len(target_neighbors)
        
        # Adamic-Adar index
        adamic_adar = 0
        for neighbor in source_neighbors.intersection(target_neighbors):
            neighbor_degree = degree_centrality[neighbor]
            if neighbor_degree > 1:
                adamic_adar += 1 / np.log(neighbor_degree)
        
        # Additional features
        degree_diff = abs(source_degree - target_degree)
        min_degree = min(source_degree, target_degree)
        max_degree = max(source_degree, target_degree)
        
        feature_vector = [
            source_degree, target_degree, source_clustering, target_clustering,
            embedding_similarity, common_neighbors, jaccard_similarity,
            preferential_attachment, adamic_adar, degree_diff, min_degree, max_degree
        ]
        
        features.append(feature_vector)
        labels.append(edge['exists'])
    
    return np.array(features), np.array(labels)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def train_link_prediction_model():
    """Train and evaluate the link prediction model"""
    
    st.info("Extracting graph features...")
    positive_edges, negative_edges, nodes = extract_graph_features()
    
    st.success(f"✅ Extracted {len(positive_edges)} positive examples and {len(negative_edges)} negative examples")
    
    st.info("Creating feature matrix...")
    X, y = create_link_prediction_features(positive_edges, negative_edges, nodes)
    
    st.success(f"✅ Created feature matrix with shape: {X.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    st.info("Training logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_names = [
        'source_degree', 'target_degree', 'source_clustering', 'target_clustering',
        'embedding_similarity', 'common_neighbors', 'jaccard_similarity',
        'preferential_attachment', 'adamic_adar', 'degree_diff', 'min_degree', 'max_degree'
    ]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    return model, scaler, feature_importance, report, roc_auc, nodes

def predict_new_links(model, scaler, nodes, top_k=10):
    """Predict new potential links in the graph"""
    
    node_names = list(nodes.keys())
    
    # Get existing edges
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        existing_edges = set()
        result = driver.execute_query("""
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN a.name as source, b.name as target
        """, database="neo4j")
        
        for record in result.records:
            existing_edges.add((record['source'], record['target']))
    
    # Sample candidate pairs (for efficiency)
    num_candidates = min(1000, len(node_names) * 10)
    candidates = []
    
    for _ in range(num_candidates):
        source = random.choice(node_names)
        target = random.choice(node_names)
        if source != target and (source, target) not in existing_edges:
            candidates.append((source, target))
    
    # Create features for candidates
    candidate_features = []
    candidate_pairs = []
    
    # We need to create dummy edges for feature calculation
    dummy_edges = [{'source': s, 'target': t, 'exists': 0} for s, t in candidates]
    positive_edges, _, _ = extract_graph_features()
    
    all_edges = positive_edges + dummy_edges
    degree_centrality, clustering_coeff = calculate_node_features(nodes, all_edges)
    
    for source, target in candidates:
        # Calculate features similar to create_link_prediction_features
        source_degree = degree_centrality[source]
        target_degree = degree_centrality[target]
        source_clustering = clustering_coeff[source]
        target_clustering = clustering_coeff[target]
        
        # Embedding similarity
        embedding_similarity = 0
        if (nodes[source]['embedding'] is not None and 
            nodes[target]['embedding'] is not None):
            try:
                source_emb = np.array(nodes[source]['embedding'])
                target_emb = np.array(nodes[target]['embedding'])
                embedding_similarity = np.dot(source_emb, target_emb) / (
                    np.linalg.norm(source_emb) * np.linalg.norm(target_emb)
                )
            except:
                embedding_similarity = 0
        
        # Calculate other features...
        source_neighbors = set()
        target_neighbors = set()
        
        for e in positive_edges:
            if e['source'] == source:
                source_neighbors.add(e['target'])
            elif e['target'] == source:
                source_neighbors.add(e['source'])
            if e['source'] == target:
                target_neighbors.add(e['target'])
            elif e['target'] == target:
                target_neighbors.add(e['source'])
        
        common_neighbors = len(source_neighbors.intersection(target_neighbors))
        union_neighbors = source_neighbors.union(target_neighbors)
        jaccard_similarity = common_neighbors / len(union_neighbors) if len(union_neighbors) > 0 else 0
        preferential_attachment = len(source_neighbors) * len(target_neighbors)
        
        adamic_adar = 0
        for neighbor in source_neighbors.intersection(target_neighbors):
            neighbor_degree = degree_centrality[neighbor]
            if neighbor_degree > 1:
                adamic_adar += 1 / np.log(neighbor_degree)
        
        degree_diff = abs(source_degree - target_degree)
        min_degree = min(source_degree, target_degree)
        max_degree = max(source_degree, target_degree)
        
        feature_vector = [
            source_degree, target_degree, source_clustering, target_clustering,
            embedding_similarity, common_neighbors, jaccard_similarity,
            preferential_attachment, adamic_adar, degree_diff, min_degree, max_degree
        ]
        
        candidate_features.append(feature_vector)
        candidate_pairs.append((source, target))
    
    # Make predictions
    X_candidates = scaler.transform(np.array(candidate_features))
    predictions_proba = model.predict_proba(X_candidates)[:, 1]
    
    # Get top predictions
    top_indices = np.argsort(predictions_proba)[-top_k:][::-1]
    
    predictions_df = pd.DataFrame({
        'Source': [candidate_pairs[i][0] for i in top_indices],
        'Target': [candidate_pairs[i][1] for i in top_indices],
        'Probability': [predictions_proba[i] for i in top_indices],
        'Rank': range(1, top_k + 1)
    })
    
    return predictions_df

def display_results(results):
    """Display the stored results"""
    if results is None:
        return
    
    model, scaler, feature_importance, report, roc_auc, nodes, predictions_df = results
    
    st.success("🎉 Link prediction analysis completed successfully!")
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>ROC AUC Score</h3>
            <h2>{roc_auc:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accuracy = report['accuracy']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2>{accuracy:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        f1_score = report['weighted avg']['f1-score']
        st.markdown(f"""
        <div class="metric-card">
            <h3>F1 Score</h3>
            <h2>{f1_score:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Top predictions
    st.subheader(f"🔮 Top {len(predictions_df)} Predicted New Links")
    
    # Create a more visually appealing predictions display
    for idx, row in predictions_df.iterrows():
        st.markdown(f"""
        <div class="prediction-card">
            <strong>#{row['Rank']}</strong> &nbsp;&nbsp;
            <strong>{row['Source']}</strong> → <strong>{row['Target']}</strong>
            <div style="float: right;">
                <span style="background: #667eea; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.9rem;">
                    {row['Probability']:.4f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature Importance Visualization
    st.subheader("Feature Importance Analysis")
    
    # Create a horizontal bar chart for feature importance
    fig = px.bar(
        feature_importance.head(10),  # Show top 10 features
        x='importance',
        y='feature',
        orientation='h',
        title="Link Prediction Features",
        color='importance',
        color_continuous_scale='viridis',
        height=500
    )
    
    fig.update_layout(
        xaxis_title="Feature Importance",
        yaxis_title="Features",
        title_x=0.5,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Downloadable results
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download predictions
        csv_predictions = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv_predictions,
            file_name="link_predictions.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download feature importance
        csv_features = feature_importance.to_csv(index=False)
        st.download_button(
            label="Download Feature Importance CSV",
            data=csv_features,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    
    # Classification report
    with st.expander("📋 Detailed Classification Report"):
        st.json(report)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    st.sidebar.title("🔧 Model Configuration")
    
    # Model parameters
    top_k = st.sidebar.slider("Number of predictions to show", 5, 20, 10)
    
    # Clear results button
    if st.sidebar.button("Clear Results", type="secondary"):
        st.session_state.ml_analysis_results = None
        st.session_state.analysis_completed = False
        st.session_state.model_cache_key = None
        st.rerun()
    
    # Connection test
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
        st.sidebar.success("✅ Connected to Neo4j")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to connect to Neo4j: {str(e)}")
        st.error("Cannot connect to Neo4j database. Please check your connection settings.")
        return
    
    # Display stored results if available
    if st.session_state.ml_analysis_results is not None:
        st.markdown("## 📊 Previous Analysis Results")
        display_results(st.session_state.ml_analysis_results)
        st.markdown("---")
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Link Prediction Analysis", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Train model
            status_text.text("Training link prediction model...")
            progress_bar.progress(20)
            
            model, scaler, feature_importance, report, roc_auc, nodes = train_link_prediction_model()
            
            progress_bar.progress(70)
            status_text.text("Generating predictions...")
            
            # Get predictions
            predictions_df = predict_new_links(model, scaler, nodes, top_k)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Store results in session state
            st.session_state.ml_analysis_results = (
                model, scaler, feature_importance, report, roc_auc, nodes, predictions_df
            )
            st.session_state.analysis_completed = True
            
            # Display results
            st.markdown("## 📊 New Analysis Results")
            display_results(st.session_state.ml_analysis_results)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.exception(e)
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    # Only show getting started section if no analysis has been completed
    elif st.session_state.ml_analysis_results is None:
        # Instructions
        st.markdown("""
        ## 🚀 Getting Started
        
        This ML Analytics tool performs **Link Prediction** on your biomedical knowledge graph using advanced machine learning techniques.
        
        ### What is Link Prediction?
        Link prediction is a fundamental task in graph analysis that aims to predict missing or future connections between nodes in a network. In the context of biomedical knowledge graphs, this can help:
        
        - 🔬 **Discover new drug-disease relationships**
        - 🧬 **Identify potential gene-protein interactions**
        - 🎯 **Suggest novel therapeutic targets**
        - 📚 **Uncover hidden research connections**
        
        Click **"Run Link Prediction Analysis"** in the sidebar to start!
        """)


if __name__ == "__main__":
    main()