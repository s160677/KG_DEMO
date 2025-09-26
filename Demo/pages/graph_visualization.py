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
if 'selected_entities' not in st.session_state:
    st.session_state.selected_entities = []
if 'initial_graph_loaded' not in st.session_state:
    st.session_state.initial_graph_loaded = False
if 'generated_cypher' not in st.session_state:
    st.session_state.generated_cypher = None

def get_available_entities():
    """Get all available entity names from the database"""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            
            # Get all entity names
            result = driver.execute_query(
                "MATCH (n) RETURN DISTINCT n.name as name ORDER BY n.name",
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.data,
            )
            
            return True, [item['name'] for item in result if item['name']], None
    except Exception as e:
        return False, [], f"❌ Failed to get entities: {str(e)}"

def get_database_schema():
    """Get database schema information for LLM context"""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            
            # Get node labels
            name_result = driver.execute_query(
                "MATCH (n) RETURN n.name as name",
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
            
            # Get sample entities and relationships for context
            sample_nodes = driver.execute_query(
                "MATCH (n) RETURN labels(n) as labels, properties(n) as properties LIMIT 10",
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.data,
            )
            
            return True, name_result, rels_result, sample_nodes, None
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

def generate_cypher_for_entities(entities):
    """Generate Cypher query for selected entities"""
    if len(entities) == 0:
        return None, "No entities selected"
    
    if len(entities) == 1:
        # Single entity - show its neighborhood
        entity = entities[0]
        query = f"""
        MATCH (e)-[r*1..3]-(neighbor)
        WHERE e.name CONTAINS '{entity}'
        RETURN e, r, neighbor
        LIMIT 50
        """
        return query, f"Showing neighborhood for: {entity}"
    
    elif len(entities) == 2:
        # Two entities - find path between them
        entity1, entity2 = entities[0], entities[1]
        query = f"""
        MATCH path = (e1)-[*1..4]-(e2)
        WHERE e1.name CONTAINS '{entity1}' AND e2.name CONTAINS '{entity2}'
        RETURN path
        LIMIT 20
        """
        return query, f"Finding path between: {entity1} and {entity2}"
    
    else:
        # Multiple entities - show all connections
        entity_conditions = " OR ".join([f"e.name CONTAINS '{entity}'" for entity in entities])
        query = f"""
        MATCH (e)-[r*1..2]-(neighbor)
        WHERE {entity_conditions}
        RETURN e, r, neighbor
        LIMIT 100
        """
        return query, f"Showing connections for: {', '.join(entities)}"

def execute_cypher_query(query, limit=100):
    """Execute Cypher query and return graph result"""
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            
            result = driver.execute_query(
                query,
                database_=NEO4J_DATABASE,
                routing_=RoutingControl.READ,
                result_transformer_=Result.graph,
            )
            return True, result, f"Query executed successfully! Found {len(result.nodes)} nodes and {len(result.relationships)} relationships."
    except Exception as e:
        return False, None, f"❌ Query failed: {str(e)}"

def extract_entities_from_cypher(cypher_query):
    """Extract entity names from Cypher query WHERE clauses"""
    import re
    
    entities = []
    
    # Pattern to match WHERE e.name CONTAINS 'entity_name'
    pattern1 = r"WHERE\s+\w+\.name\s+CONTAINS\s+'([^']+)'"
    matches1 = re.findall(pattern1, cypher_query, re.IGNORECASE)
    entities.extend(matches1)
    
    # Pattern to match WHERE e1.name CONTAINS 'entity1' AND e2.name CONTAINS 'entity2'
    pattern2 = r"AND\s+\w+\.name\s+CONTAINS\s+'([^']+)'"
    matches2 = re.findall(pattern2, cypher_query, re.IGNORECASE)
    entities.extend(matches2)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    return unique_entities

def create_visualization_graph(neo4j_result, requested_entities=None) -> VisualizationGraph:
    """Create visualization graph from Neo4j result"""
    try:
        VG = from_neo4j(neo4j_result)

        # Set node captions
        original_captions = []

        for node in VG.nodes:
            node.caption = node.properties.get("name") or ", ".join(node.labels)
            original_captions.append(node.caption)

        if requested_entities:
            for node in VG.nodes:
                if not any(entity.lower() in node.caption.lower() for entity in requested_entities):
                    node.caption = "other_nodes"
            VG.color_nodes(field="caption")
            
            # revert the caption to the original caption
            for node, caption in zip(VG.nodes, original_captions):
                node.caption = caption

        return VG

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None


def render_graph(VG: VisualizationGraph, height: int, initial_zoom: float = 0.1) -> HTML:
    """Render the visualization graph"""
    return VG.render(initial_zoom=initial_zoom, height=f"{height}px")


########################################################
########################################################
########################################################


# Main app
st.title("🔗 Biomedical Knowledge Graph")
st.markdown("Interactive graph visualization powered by Streamlit and Neo4J")

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

# Ensure default exists in session_state
if "selected_entities" not in st.session_state:
    st.session_state.selected_entities = []

with st.sidebar:
    st.header("Controls")
    
    # Entity Selection
    st.subheader("Entity Selection")
    
    # Get available entities
    with st.spinner("Loading available entities..."):
        success, entity_names, error = get_available_entities()
    
    if not success:
        st.error(error)
        entity_names = []
    
    # Entity selection
    selected_entities = st.multiselect(
        "Select one or more entities to visualize:",
        entity_names,
        help="Select entities to explore their relationships"
    )
    
    # Update session state
    st.session_state.selected_entities = selected_entities
    
    # Query execution
    if st.button("🔍 Execute Query", type="primary"):
        if selected_entities:
            with st.spinner("Generating and executing query..."):
                query, description = generate_cypher_for_entities(selected_entities)
                
                if query:
                    st.success(f"✅ {description}")
                    st.session_state.generated_cypher = query
                    st.code(query, language="cypher")
                    
                    # Execute the query
                    exec_success, result, exec_message = execute_cypher_query(query, 100)
                    
                    if exec_success:
                        st.session_state.query_results = result
                        st.success(exec_message)
                        
                        # Create visualization graph
                        requested_entities = extract_entities_from_cypher(query)
                        VG = create_visualization_graph(result, requested_entities)
                        if VG:
                            st.session_state.visualization_graph = VG
                            st.success("✅ Visualization created successfully!")
                        else:
                            st.error("❌ Failed to create visualization")
                    else:
                        st.error(exec_message)
                        # Check if it's a "no link found" case
                        if "no link found" in exec_message.lower() or len(result.nodes) == 0:
                            st.info("🔍 No direct connections found between the selected entities. Try selecting different entities or check if they exist in the database.")
        else:
            st.warning("Please select at least one entity!")
    
    # Reset button
    if st.button("🔄 Reset to Initial View"):
        st.session_state.selected_entities = []
        st.session_state.query_results = None
        st.session_state.visualization_graph = None
        st.session_state.initial_graph_loaded = False
        st.rerun()
    
    st.divider()
    
    # Visualization controls
    st.subheader("Visualization Settings")
    height = st.slider("Height in pixels", min_value=100, max_value=1000, value=700, step=50)
    initial_zoom = st.slider("Initial Zoom", min_value=0.1, max_value=2.0, value=0.1, step=0.1)

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
    else:
        st.info("Select entities from the sidebar to see the visualization")

with col2:
    st.header("Graph Statistics")
    
    if st.session_state.query_results:
        result = st.session_state.query_results
        
        # Basic stats
        with st.container():
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Nodes", len(result.nodes))
            with col_b:
                st.metric("Relationships", len(result.relationships))
        
        # Node label distribution
        if result.nodes:
            labels = []
            for node in result.nodes:
                for item in node.items():
                    if item[0] == "name":
                        labels.append(item[1])
            
            if labels:
                with st.expander("🏷️ Node Labels", expanded=False):
                    # Create a scrollable container for node labels
                    with st.container():
                        # Use columns to create a more compact layout
                        for label in sorted(labels):
                            st.write(f"• {label}")
        
        # Relationship type distribution
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
        
        # Summary
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
    <p>Neo4j Graph Visualization Dashboard | Powered by Streamlit & Neo4J</p>
</div>
""", unsafe_allow_html=True)