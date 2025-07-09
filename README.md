# LATTE ☕ — LLM-Assisted Topic/Thematic Analysis

The goal of this project is to assist with analysing the common themes or topics within large quantities of short texts while preserving researcher flexibility and control.

The project combines ideas from traditional computational approaches to topic analysis (e.g BERTopic), cluster annotations by LLMs, and visualisation allowing researchers to further validate and analyse results qualitatively.

For demonstration, we use Stack Exchange datasets from various communities.

## Citation

If you use LATTE in your research, please cite:

```
[Citation will be added once preprint is published]
```

For now, you can reference this repository:
```
@misc{latte2025,
  title={LATTE: LLM-Assisted Topic/Thematic Analysis},
  author={[Ivan Smirnov]},
  year={2025},
  url={https://github.com/aivan-au/latte}
}
```

# Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setting up the environment

1. Clone the repository:
   ```
   git clone https://github.com/aivan-au/latte
   cd latte
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Dataset Setup

1. Download the Stack Exchange datasets from Internet Archive:
   ```
   https://archive.org/details/stackexchange_20240630_json
   ```
2. Place the `*.7z` files in the `data/` directory
3. Process the Stack Exchange data into CSV format:
   ```
   python scripts/process_stackexchange_data.py
   ```
   This creates separate CSV files for each dataset (e.g., `data/biology.csv`) with title, text, project, and tags columns.

4. Sample datasets:
   ```
   python scripts/sample_csv_files.py
   ```
   This randomly samples 1000 rows from each CSV file and saves them with a `_sample.csv` suffix (e.g., `biology_sample.csv`).

### Computing Embeddings

5. **Compute embeddings for all datasets** (one-time setup):
   ```bash
   python scripts/compute_embeddings.py
   ```
   This automatically generates embeddings using selected methods (Hugging Face, OpenAI, Gemini) and saves them as `.pkl` files in the `embeddings/` directory. Once computed, embeddings can be reused without recomputing.

6. **Load embeddings and analyse**:
   ```python
   df = pd.read_csv('data/biology_sample.csv')
   latte = Latte(df)
   latte.embed('file', embeddings_file='embeddings/biology_minilm.pkl').reduce().plot()
   ```

### Clustering Analysis

7. **Perform clustering analysis**:
   ```python
   # Load data and embeddings
   df = pd.read_csv('data/biology_sample.csv')
   latte = Latte(df)
   
   # Embed, reduce dimensions, and cluster
   latte.embed('file', embeddings_file='embeddings/biology_minilm.pkl').reduce().cluster()
   
   # Visualize clusters at a certain granularity level
   latte.plot(cluster_level=2)
   ```

### LLM-Assisted Cluster Annotation

8. **Add LLM-generated annotations to clusters**:
   ```python
   # After clustering, add annotations using different LLM providers
   
   # Use Hugging Face models (default: Qwen/Qwen2.5-7B-Instruct)
   # By default, annotates clusters across all levels
   latte.annotate(method='hf')
   
   # Force MPS acceleration on Mac (faster on Apple Silicon)
   latte.annotate(method='hf', device='mps')
   
   # Use OpenAI API (requires OPENAI_API_KEY environment variable)
   latte.annotate(method='openai', model_name='gpt-4.1-nano')
   
   # Use Google Gemini API (requires GOOGLE_API_KEY environment variable)
   latte.annotate(method='gemini', model_name='gemini-1.5-flash')
   
   # Use custom annotation function
   def my_annotator(titles):
       return "Custom annotation based on titles"
   latte.annotate(custom_function=my_annotator)
   
   # Annotate only clusters at a specific level
   latte.annotate(method='openai', level=0)  # Only level 0 clusters
   
   # Print clusters with annotations
   latte.print_clusters()
   
   # Clear annotation model cache to free memory (optional)
   latte.clear_annotation_cache()
   
   # Export complete cluster structure to JSON
   latte.export('results.json')
   
   # Export with binary mask analysis (e.g., analyzing posts with specific tags)
   # Create a binary mask for posts containing 'biochemistry' tag
   mask = [1 if 'biochemistry' in tags else 0 for tags in df['tags']]
   latte.export('results_with_mask.json', mask=mask)
   ```

## Interactive Visualization

Exported JSON data can be visualized using the interactive browser-based viewer located in `viz/viewer.html`. See an example at [https://aivan.au/latte/](https://aivan.au/latte/).

### Local Development
1. Open `viz/viewer.html` in a web browser
2. Load your LATTE JSON export using the file input
3. Explore your data with interactive features:
   - **Search**: Find documents containing specific terms
   - **Level filtering**: Adjust cluster granularity  
   - **Mask analysis**: Compare subsets of your data
   - **Annotations**: View LLM-generated cluster descriptions

### Web Deployment

To share your visualizations online, use the deployment script:

```bash
# Basic deployment
python scripts/deploy_visualization.py data/results.json

# Custom name and single-file deployment
python scripts/deploy_visualization.py data/results.json --name my_analysis --inline

# Create a zip package for easy sharing
python scripts/deploy_visualization.py data/results.json --name my_analysis --zip
```

**Deployment options:**
- `--name`: Custom name for deployment folder (default: `latte_visualization`)
- `--inline`: Create single standalone HTML file with embedded CSS/JS
- `--zip`: Package deployment as ZIP file for easy distribution

**Output:**
- **Regular deployment**: Upload entire `deployments/[name]/` folder to web server
- **Inline deployment**: Upload only `deployments/[name]/index.html` file
- **Zip deployment**: Extract and upload `deployments/[name].zip` contents

For details on Embeddings, see [Embeddings Guide](EMBEDDINGS.md) and [Embeddings Demo](demo/Embeddings%20Demo.ipynb). 
For more clustering examples and parameter options, see [Clustering Demo](demo/Clustering%20Demo.ipynb).
For annotation examples and LLM integration, see [Annotation Demo](demo/Annotation%20Demo.ipynb)