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

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Collection name
COLLECTION_NAME = "image_search_python_streamlit"

# Initialize models
text_embedding = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
image_embedding = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")

def search_images(query, limit=5):
    """Search for images based on text query"""
    # Generate query embedding
    query_embedding = next(text_embedding.embed(query))
    
    # Search in Qdrant
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=limit
    )
    
    return search_result

# Streamlit UI
st.title("Housing Multi-Modal Similarity Search")

# Search interface
query = st.text_input("Enter your search query:")
if query:
    results = search_images(query)
    
    # Display results
    st.subheader("Search Results")
    cols = st.columns(3)
    for i, result in enumerate(results):
        with cols[i % 3]:
            st.image(result.payload["path"], use_column_width=True)
            st.write(f"Score: {result.score:.2f}") 