import streamlit as st
import pandas as pd
from neo4j import GraphDatabase, Result, RoutingControl
from neo4j_viz.neo4j import from_neo4j
from neo4j_viz import VisualizationGraph
import streamlit.components.v1 as components
from IPython.display import HTML
import openai
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Visualization with AI",
    page_icon="",
    layout="wide"
)

# Neo4j connection configuration
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_DATABASE = st.secrets["NEO4J_DATABASE"]

# Initialize session state
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'visualization_graph' not in st.session_state:
    st.session_state.visualization_graph = None
if 'generated_cypher' not in st.session_state:
    st.session_state.generated_cypher = ""
if 'last_natural_query' not in st.session_state:
    st.session_state.last_natural_query = ""
if 'initial_graph_loaded' not in st.session_state:
    st.session_state.initial_graph_loaded = False


def get_database_schema():
    """Get database schema information for LLM context"""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            
            # Get node labels
            labels_result = driver.execute_query(
                "CALL db.labels()",
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.data,
            )
            
            # Get relationship types
            rels_result = driver.execute_query(
                "CALL db.relationshipTypes()",
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.data,
            )
            
            # Get sample nodes and relationships for context
            sample_nodes = driver.execute_query(
                "MATCH (n) RETURN labels(n) as labels, keys(n) as properties LIMIT 10",
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.data,
            )
            
            return True, labels_result, rels_result, sample_nodes, None
    except Exception as e:
        return False, None, None, None, f"❌ Schema query failed: {str(e)}"

def load_initial_graph():
    """Load the entire graph for initial visualization"""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            
            # Query to get a representative sample of the entire graph
            # This query gets nodes and their immediate relationships
            query = """
            MATCH (n)-[r]-(m)
            RETURN n, r, m
            """
            
            result = driver.execute_query(
                query,
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.graph,
            )
            
            return True, result, f"Initial graph loaded! Found {len(result.nodes)} nodes and {len(result.relationships)} relationships."
    except Exception as e:
        return False, None, f"❌ Failed to load initial graph: {str(e)}"


def generate_cypher_query(natural_language_query, schema_info):
    """Use LLM to generate Cypher query from natural language"""
    try:
        # Get schema information
        success, labels, rels, sample_data, error = get_database_schema()
        
        if not success:
            return False, None, error
        
        # Build schema context
        labels_list = [label['label'] for label in labels] if labels else []
        rels_list = [rel['relationshipType'] for rel in rels] if rels else []
        
        # Create schema context
        schema_context = f"""
                        Database Schema:
                        - Node Labels: {', '.join(labels_list)}
                        - Relationship Types: {', '.join(rels_list)}
                        - Sample Properties: {sample_data[:3] if sample_data else 'No sample data'}
                        """
        
        # Create the prompt
        prompt = f"""
                You are a Neo4j Cypher query expert. Convert the following natural language query to a valid Cypher query.

                Database Schema:
                {schema_context}

                Natural Language Query: "{natural_language_query}"

                When user asks for the neighborhood for one entity (ex. thymosin),use the following template to generate the query:
                ```
                MATCH (e:Entity)-[r*1..3]-(neighbor)
                WHERE e.name CONTAINS 'thymosin'
                RETURN e, neighbor, r
                LIMIT 30
                ```

                When user asks for the neighborhood between multiple entities (ex. thymosin and prothymosin alpha), use the following template to generate the query:
                ```
                MATCH (e1:Entity)-[r*1..3]-(e2:Entity)
                WHERE e1.name CONTAINS 'thymosin' AND e2.name CONTAINS 'prothymosin alpha'
                RETURN e1, r1, e2
                LIMIT 30
                ```

                Instructions:
                1. Return ONLY the Cypher query, no explanations
                2. Use LIMIT 30 to keep results manageable unless the user specifies a different limit
                3. Use proper Cypher syntax
                4. If the query is ambiguous, make reasonable assumptions based on the schema
                5. Focus on returning nodes and relationships that can be visualized

                Cypher Query:
                """
        
        # Use OpenAI API (you can replace with other LLMs)
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a Neo4j Cypher expert. Return only valid Cypher queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        cypher_query = response.choices[0].message.content.strip()
        
        # Clean up the response (remove any markdown formatting)
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.split("```")[1].strip()
        if cypher_query.startswith("cypher"):
            cypher_query = cypher_query.replace("cypher", "").strip()
        
        return True, cypher_query, "✅ Cypher query generated successfully!"
        
    except Exception as e:
        return False, None, f"❌ LLM query generation failed: {str(e)}"

def execute_cypher_query(query, limit=100):
    """Execute Cypher query and return graph result"""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            
            # Add LIMIT if not present and query doesn't already have one
            if "LIMIT" not in query.upper() and "RETURN" in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            result = driver.execute_query(
                query,
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.graph,
            )
            return True, result, f"Query executed successfully! Found {len(result.nodes)} nodes and {len(result.relationships)} relationships."
    except Exception as e:
        return False, None, f"❌ Query failed: {str(e)}"

def create_visualization_graph(neo4j_result) -> VisualizationGraph:
    """Create visualization graph from Neo4j result"""
    try:
        VG = from_neo4j(neo4j_result)

        # set node label property to name
        for node in VG.nodes:
            node.caption = node.properties.get("name") or ", ".join(node.labels)
        return VG
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def render_graph(VG: VisualizationGraph, height: int, initial_zoom: float = 0.1) -> HTML:
    """Render the visualization graph"""
    return VG.render(initial_zoom=initial_zoom, height=f"{height}px")


########################################################
########################################################
########################################################


# Main app
st.title(" Biomedical Knowledge Graph")
st.markdown("Interactive graph visualization powered by Streamlit, LLM, and Neo4J")

# Load initial graph if not already loaded
if not st.session_state.initial_graph_loaded:
    with st.spinner("Loading initial graph visualization..."):
        success, result, message = load_initial_graph()
        if success:
            st.session_state.query_results = result
            VG = create_visualization_graph(result)
            if VG:
                st.session_state.visualization_graph = VG
                st.session_state.initial_graph_loaded = True
            else:
                st.error("❌ Failed to create initial visualization")
        else:
            st.error(message)

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Natural Language Query
    st.subheader(" Natural Language Query")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        st.info("You can get an API key from: https://platform.openai.com/api-keys")
    
    natural_query = st.text_area(
        "Describe what you want to find:",
        value=st.session_state.last_natural_query,
        height=100,
        help="Describe what you want to find in the graph"
    )
    
    # Generate and Execute button
    if st.button("🧠 Generate & Execute Query", type="primary"):
        if natural_query.strip():
            st.session_state.last_natural_query = natural_query
            with st.spinner("Generating Cypher query..."):
                success, cypher_query, message = generate_cypher_query(natural_query, None)
                
                if success:
                    st.session_state.generated_cypher = cypher_query
                    st.success(message)
                    st.code(cypher_query, language="cypher")
                    
                    # Automatically execute the generated query
                    with st.spinner("Executing query..."):
                        exec_success, result, exec_message = execute_cypher_query(cypher_query, 100)
                        
                        if exec_success:
                            st.session_state.query_results = result
                            st.success(exec_message)
                            
                            # Create visualization graph
                            VG = create_visualization_graph(result)
                            if VG:
                                st.session_state.visualization_graph = VG
                                st.success("✅ Visualization created successfully!")
                            else:
                                st.error("❌ Failed to create visualization")
                        else:
                            st.error(exec_message)
                else:
                    st.error(message)
        else:
            st.warning("Please enter a natural language query!")
    
    st.divider()
    
    # Visualization controls
    st.subheader("Visualization Settings")
    # Default values
    height = st.slider("Height in pixels", min_value=100, max_value=1000, value=700, step=50)
    initial_zoom = st.slider("Initial Zoom", min_value=1.0, max_value=2.0, value=0.1, step=0.1)
    
# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Graph Visualization")
    
    if st.session_state.visualization_graph:
        VG = st.session_state.visualization_graph
        
        # Display the visualization
        components.html(
            render_graph(VG, height=height, initial_zoom=initial_zoom).data,
            height=height,
        )
        
        # Show graph info
        if st.session_state.query_results:
            result = st.session_state.query_results
    else:
        st.info("Enter a natural language query to see the visualization")

with col2:
    st.header("Graph Statistics")
    
    if st.session_state.query_results:
        result = st.session_state.query_results
        
        # Basic stats in a container
        with st.container():
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Nodes", len(result.nodes))
            with col_b:
                st.metric("Relationships", len(result.relationships))
        
        # Node label distribution in expandable section
        if result.nodes:
            labels = {}
            for node in result.nodes:
                for label in node.labels:
                    labels[label] = labels.get(label, 0) + 1
            
            if labels:
                with st.expander("🏷️ Node Labels", expanded=False):
                    # Create a scrollable container for node labels
                    with st.container():
                        # Use columns to create a more compact layout
                        for label, count in sorted(labels.items()):
                            st.write(f"• **{label}**: {count}")
        
        # Relationship type distribution in expandable section
        if result.relationships:
            rel_types = {}
            for rel in result.relationships:
                rel_type = rel.type
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            if rel_types:
                with st.expander("🔗 Relationship Types", expanded=False):
                    # Create a scrollable container for relationship types
                    with st.container():
                        # Display relationship types in a single column
                        for rel_type, count in sorted(rel_types.items()):
                            st.write(f"• **{rel_type}**: {count}")
        
        # Add a summary section
        with st.expander("📈 Summary", expanded=False):
            total_nodes = len(result.nodes)
            total_rels = len(result.relationships)
            avg_connections = total_rels / total_nodes if total_nodes > 0 else 0
            
            st.write(f"**Total Elements**: {total_nodes + total_rels}")
            st.write(f"**Average Connections per Node**: {avg_connections:.2f}")
            
            if result.nodes:
                unique_labels = len(set(label for node in result.nodes for label in node.labels))
                st.write(f"**Unique Node Types**: {unique_labels}")
            
            if result.relationships:
                unique_rel_types = len(set(rel.type for rel in result.relationships))
                st.write(f"**Unique Relationship Types**: {unique_rel_types}")
    else:
        st.info("No query results to display")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Neo4j Graph Visualization Dashboard | Powered by Streamlit, LLM & Neo4J</p>
</div>
""", unsafe_allow_html=True)