"""
Annotation methods for the Latte class.

This module contains the implementation details for different LLM providers:
- Hugging Face transformers
- OpenAI API
- Google Gemini API

These methods are separated from the main Latte class to keep the core logic clean
and modular.
"""

import os
import random
import time
from dotenv import load_dotenv

# Global cache for models to avoid reloading
_model_cache = {}

def annotate_with_hf(titles, model_name="Qwen/Qwen2.5-7B-Instruct", system_prompt=None, device=None):
    """Generate cluster annotation using Hugging Face transformers model."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        raise ImportError("transformers, torch, and accelerate are required for Hugging Face annotations. Install with: pip install transformers torch accelerate")
    
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that identifies the main topic or key topics from a set of texts. You must provide only the main topics in a single, concise sentence. Do not add any introductions or explanations. Do not start with phrases like 'The texts cover', 'The texts focus on', 'These texts are about', or similar introductory phrases. Just state the topic directly."
    
    # Determine the best device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA for acceleration")
        else:
            device = "cpu"
            print("Using CPU (no GPU acceleration available)")
    else:
        print(f"Using specified device: {device}")
    
    # Check if model is already cached
    cache_key = f"hf_{model_name}_{device}"
    if cache_key not in _model_cache:
        print(f"Loading Hugging Face model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with better parameters
        if device == "mps":
            # For MPS, we need to be more specific about device placement
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # MPS works well with float16
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        else:
            # For CUDA or CPU, use auto device mapping
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if device != "cpu" else None,
                low_cpu_mem_usage=True
            )
        
        _model_cache[cache_key] = {
            'model': model,
            'tokenizer': tokenizer
        }
        print(f"Model {model_name} loaded and cached")
    else:
        print(f"Using cached model: {model_name}")
    
    # Get cached model and tokenizer
    model = _model_cache[cache_key]['model']
    tokenizer = _model_cache[cache_key]['tokenizer']
    
    # Prepare the prompt
    titles_text = "\n".join([f"- {title}" for title in titles])
    user_prompt = f"Here are the titles from a cluster of related texts:\n\n{titles_text}\n\nWhat is the main topic or theme of this cluster?"
    
    # Format the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Prepare model inputs
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    
    # Generate response
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    inference_time = time.time() - start_time
    print(f"Inference took {inference_time:.2f} seconds")
    
    return response.strip()


def annotate_with_openai(titles, model_name="gpt-4.1-nano", system_prompt=None):
    """Generate cluster annotation using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required for OpenAI annotations. Install with: pip install openai")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that identifies the main topic or key topics from a set of texts. You must provide only the main topics in a single, concise sentence. Do not add any introductions or explanations. Do not start with phrases like 'The texts cover', 'The texts focus on', 'These texts are about', or similar introductory phrases. Just state the topic directly."
    
    client = OpenAI(api_key=api_key)
    
    # Prepare the prompt
    titles_text = "\n".join([f"- {title}" for title in titles])
    user_prompt = f"Here are the titles from a cluster of related texts:\n\n{titles_text}\n\nWhat is the main topic or theme of this cluster?"
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()


def annotate_with_gemini(titles, model_name="gemini-1.5-flash", system_prompt=None):
    """Generate cluster annotation using Google Gemini API."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai is required for Gemini annotations. Install with: pip install google-genai")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that identifies the main topic or key topics from a set of texts. You must provide only the main topics in a single, concise sentence. Do not add any introductions or explanations. Do not start with phrases like 'The texts cover', 'The texts focus on', 'These texts are about', or similar introductory phrases. Just state the topic directly."
    
    # Initialize the client
    client = genai.Client(api_key=api_key)
    
    # Prepare the prompt
    titles_text = "\n".join([f"- {title}" for title in titles])
    user_prompt = f"Here are the titles from a cluster of related texts:\n\n{titles_text}\n\nWhat is the main topic or theme of this cluster?"
    
    # Combine system and user prompts
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=100,
            temperature=0.7
        )
    )
    
    return response.text.strip()


def clear_model_cache():
    """Clear the model cache to free up memory."""
    global _model_cache
    _model_cache.clear()
    print("Model cache cleared") 