# Embeddings

## Overview

LATTE supports multiple embedding methods to convert your texts into numerical representations. Embeddings only need to be computed once and can be saved/reused for future analysis.

## Built-in Methods

### Hugging Face (Default)
Local transformer models, no API key required:
```python
# Default model (all-MiniLM-L6-v2)
latte.embed()

# Specify different model
latte.embed(method='hf', model_name='all-mpnet-base-v2')
```

### OpenAI
High-quality embeddings via API (requires `OPENAI_API_KEY`):
```python
latte.embed(method='openai')
```

### Google Gemini
Alternative API option (requires `GOOGLE_API_KEY`):
```python
latte.embed(method='gemini')
```

## Custom Embeddings

You can provide your own embedding function:
```python
def my_embeddings(texts):
    # Your custom embedding logic here
    return embeddings_array

latte.embed(custom_function=my_embeddings)
```

## Saving and Loading Embeddings

### Save embeddings during computation:
```python
latte.embed(method='hf', save_to_file='my_embeddings.pkl')
```

### Load pre-computed embeddings:
```python
latte.embed(method='file', embeddings_file='my_embeddings.pkl')
```

## File Formats

LATTE supports two embedding file formats:
- **`.pkl`**: Dictionary mapping texts to embeddings
- **`.npy`**: Numpy array (must match text order)

## API Setup

For OpenAI and Google Gemini methods, you'll need API keys. Create a `.env` file in your project root:

```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

**Getting API Keys:**
- **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com) and generate an API key
- **Google Gemini**: Create a project at [Google AI Studio](https://aistudio.google.com) and get your API key

**Note:** Hugging Face models run locally and don't require API keys. 