"""
Embedding computation methods for the Latte class.

This module contains the implementation details for different embedding providers:
- Hugging Face sentence transformers
- OpenAI embeddings API  
- Google Gemini embeddings API

These methods are separated from the main Latte class to keep the core logic clean
and modular.
"""

import numpy as np
import os
from dotenv import load_dotenv


def compute_hf_embeddings(texts, model_name):
    """Compute embeddings using Hugging Face sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence_transformers is required for Hugging Face embeddings. Install with: pip install sentence-transformers")
    
    print(f"Loading Hugging Face model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def compute_openai_embeddings(texts):
    """Compute embeddings using OpenAI API."""
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required for OpenAI embeddings. Install with: pip install openai")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    
    print("Computing OpenAI embeddings...")
    embeddings = []
    
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        
        batch_embeddings = [embedding.embedding for embedding in response.data]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def compute_gemini_embeddings(texts):
    """Compute embeddings using Google Gemini API."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai is required for Gemini embeddings. Install with: pip install google-genai")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    # Initialize the client
    client = genai.Client(api_key=api_key)
    
    print("Computing Gemini embeddings...")
    embeddings = []
    
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Create config for the embedding request
        config = types.EmbedContentConfig(task_type="CLUSTERING")
        
        result = client.models.embed_content(
            model="text-embedding-004",
            #model="gemini-embedding-exp-03-07",
            contents=batch,
            config=config
        )
        
        batch_embeddings = [emb.values for emb in result.embeddings]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings) 