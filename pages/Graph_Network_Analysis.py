import streamlit as st
import pandas as pd
import networkx as nx
from itertools import combinations
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os


# Function to create a graph based on sales data for a specific country
def create_sales_graph(data, country):
    filtered_data = data[data['sales_country'] == country]
    G = nx.Graph()
    for _, row in filtered_data.iterrows():
        G.add_node(row['customer_id'], node_type='customer', title=row['customer_id'])
        G.add_node(row['material_id'], node_type='material', title=row['material_id'])
        G.add_edge(row['customer_id'], row['material_id'])
    return G


# Function to visualize the graph using pyvis
def visualize_graph(G):
    nt = Network(notebook=True, height="950px", width="100%")
    nt.from_nx(G)
    # Customize node colors based on node type
    for node in nt.nodes:
        if node["title"].startswith("CUSID"):
            node["color"] = "#ff9999"
        else:
            node["color"] = "#99ff99"

    # Adjust physics settings to make the graph movement slower and more stable
    nt.options = {
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.1
            },
            "minVelocity": 0.75,
            "solver": "barnesHut",
            "timestep": 0.5
        }
    }

    # Create the temp directory if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Generate HTML and visualize
    # Use tempfile.mkstemp to create a temporary file within the specified directory
    fd, temp_path = tempfile.mkstemp(suffix=".html", dir=temp_dir)
    os.close(fd)  # Close the file descriptor to avoid leaking it
    nt.save_graph(temp_path)
    return temp_path


# Function to find similar customers based on Jaccard coefficient
def find_similar_customers(G, customer_id):
    jaccard_coefficients = [(u, v, p) for u, v, p in nx.jaccard_coefficient(G, [(customer_id, x) for x in G.nodes if
                                                                                G.nodes[x]['node_type'] == 'customer' and x != customer_id])]
    similar_customers = sorted(jaccard_coefficients, key=lambda x: x[2], reverse=True)
    return similar_customers


# Streamlit app
def main():
    st.title("Customer Purchasing Behavior Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='sales')

        # Select sales country
        country = st.selectbox('Select a sales country:', df['sales_country'].unique())

        # Create graph
        G = create_sales_graph(df, country)

        # Visualize the graph
        try:
            html_file = visualize_graph(G)
            st.markdown(f'<a href="{html_file}" target="_blank">Open Graph in New Tab</a>', unsafe_allow_html=True)
            HtmlFile = open(html_file, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=800)
        except Exception as e:
            st.write("An error occurred in graph visualization:", e)

        # Select customer ID
        customer_id = st.selectbox('Select a customer ID:',list(set(df[df['sales_country'] == country]['customer_id'])))

        # Find similar customers
        similar_customers = find_similar_customers(G, customer_id)

        st.write("Similar customers based on purchasing behavior: [customer] --> [similarity score]")
        for sim in similar_customers:
            st.write(f"{sim[1]} --> {sim[2]:.3f}")


if __name__ == "__main__":
    main()

