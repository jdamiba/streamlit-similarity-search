import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding, ImageEmbedding
from fastembed.common.model_management import disable_progress_bars
from PIL import Image
import glob
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Disable progress bars
disable_progress_bars()

# Get Qdrant credentials
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Verify credentials
if not qdrant_url or not qdrant_api_key:
    st.error("Missing Qdrant credentials. Please check your .env file.")
    st.stop()

# Initialize Qdrant client with error handling
try:
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    # Test the connection
    client.get_collections()
except Exception as e:
    st.error(f"Failed to connect to Qdrant: {str(e)}")
    st.error(f"URL: {qdrant_url}")
    st.error("Please verify your Qdrant URL and API key are correct.")
    st.stop()

# Collection name
COLLECTION_NAME = "image_search_python_streamlit"

# Initialize models with error handling
try:
    # Using CLIP for both text and image embeddings
    text_embedding = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
    image_embedding = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
except Exception as e:
    st.error(f"Failed to initialize models: {str(e)}")
    st.error("Please try again later or contact support if the issue persists.")
    st.stop()

def verify_collection():
    """Verify that the collection exists and is accessible"""
    try:
        collections = client.get_collections()
        available_collections = [col.name for col in collections.collections]
        st.write("Available collections:", available_collections)
        
        if COLLECTION_NAME not in available_collections:
            st.error(f"Collection '{COLLECTION_NAME}' not found. Available collections are: {available_collections}")
            return False
        return True
    except Exception as e:
        st.error(f"Error verifying collections: {str(e)}")
        return False

def search_images(query, limit=5):
    """Search for images based on text query"""
    try:
        # Generate query embedding
        query_embedding = next(text_embedding.embed(query))
        
        # Search in Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        return search_result
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return None

# Streamlit UI
st.title("Housing Multi-Modal Similarity Search")

# Verify collection exists
if not verify_collection():
    st.stop()

# Search interface
query = st.text_input("Enter your search query:")
if query:
    results = search_images(query)
    
    if results is None:
        st.error("Search failed. Please try again.")
    elif len(results) == 0:
        st.info("No results found for your query.")
    else:
        # Display results
        st.subheader("Search Results")
        cols = st.columns(3)
        for i, result in enumerate(results):
            with cols[i % 3]:
                st.image(result.payload["path"], use_column_width=True)
                st.write(f"Score: {result.score:.2f}") 