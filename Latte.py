import pandas as pd
import numpy as np
import os
import pickle
import warnings
import random
from embedding_methods import compute_hf_embeddings, compute_openai_embeddings, compute_gemini_embeddings
from annotation_methods import annotate_with_hf, annotate_with_openai, annotate_with_gemini, clear_model_cache
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

# Suppress sklearn deprecation warning about force_all_finite -> ensure_all_finite
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

class Latte:
    """
    LATTE ☕ — LLM-Assisted Topic/Thematic Analysis
    
    A class for analyzing common themes or topics within large quantities of short texts
    while preserving researcher flexibility and control.
    """
    
    def __init__(self, data, title_field='title', text_field='text', 
                 include_title_in_text=True, mute=False):
        """
        Initialize the Latte analyzer.
        
        Args:
            data (pd.DataFrame): The input data frame containing the text data
            title_field (str): Name of the column containing titles (used for output/listing)
            text_field (str): Name of the column containing the main text content
            include_title_in_text (bool): If True, combines title + newline + text for embeddings
            mute (bool): If True, suppresses all print statements from the class (default: False)
        """
        self.df = data.copy()
        self.title_field = title_field
        self.text_field = text_field
        self.include_title_in_text = include_title_in_text
        self.mute = mute
        
        # Extract titles for easy access (used for output/listing)
        self.titles = self.df[title_field].tolist()
        
        # Extract and prepare texts for embeddings
        if include_title_in_text:
            # Combine title + newline + text for full text embeddings
            self.texts = (self.df[title_field].astype(str) + '\n' + 
                         self.df[text_field].astype(str)).tolist()
        else:
            # Use only the text field for embeddings
            self.texts = self.df[text_field].tolist()
        
        # Initialize embeddings placeholder
        self.embeddings = None
        
        self._print(f"Initialized Latte with {len(self.df)} documents")

    def _print(self, *args, **kwargs):
        """Internal print method that respects the mute setting."""
        if not self.mute:
            print(*args, **kwargs)

    def embed(self, method='hf', model_name='all-MiniLM-L6-v2', embeddings_file=None, 
              save_to_file=None, custom_function=None):
        """
        Compute and store text embeddings using various methods.
        
        Args:
            method (str): Embedding method - 'hf', 'openai', 'gemini', 'file', or 'custom'
            model_name (str): Model name for Hugging Face embeddings (default: 'all-MiniLM-L6-v2')
            embeddings_file (str): Path to file containing pre-computed embeddings (for method='file')
            save_to_file (str): Optional path to save computed embeddings for future use
            custom_function (callable): Custom embedding function that takes a list of texts and returns embeddings
        
        Returns:
            self: Returns self for method chaining
            
        Examples:
            # Use built-in methods
            latte.embed(method='hf', model_name='all-MiniLM-L6-v2')
            latte.embed(method='openai')
            latte.embed(method='gemini')
            
            # Use custom function
            def my_embedding_function(texts):
                # Your custom embedding logic here
                return embeddings_array
            
            latte.embed(method='custom', custom_function=my_embedding_function)
            
            # Or simply pass the function directly
            latte.embed(custom_function=my_embedding_function)
        """
        # If custom_function is provided, use it regardless of method
        if custom_function is not None:
            if not callable(custom_function):
                raise ValueError("custom_function must be a callable that takes a list of texts and returns embeddings")
            
            self._print("Computing embeddings using custom function...")
            self.embeddings = custom_function(self.texts)
            
        elif method == 'custom':
            if custom_function is None:
                raise ValueError("custom_function must be provided when method='custom'")
            
        elif method == 'file':
            self.embeddings = self._load_embeddings_from_file(embeddings_file)
            
        elif method == 'hf':
            self.embeddings = compute_hf_embeddings(self.texts, model_name)
            
        elif method == 'openai':
            self.embeddings = compute_openai_embeddings(self.texts)
            
        elif method == 'gemini':
            self.embeddings = compute_gemini_embeddings(self.texts)
            
        else:
            raise ValueError(f"Unknown embedding method: {method}. Available methods: 'hf', 'openai', 'gemini', 'file', 'custom'")
        
        self._print(f"Embeddings computed successfully. Shape: {self.embeddings.shape}")
        
        # Save embeddings to file if requested
        if save_to_file:
            self._save_embeddings_to_file(save_to_file)
        
        return self
    
    def _load_embeddings_from_file(self, embeddings_file):
        """Load pre-computed embeddings from file."""
        if not embeddings_file:
            raise ValueError("embeddings_file must be provided when method='file'")
        
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        self._print(f"Loading embeddings from: {embeddings_file}")
        
        # Determine file format and load accordingly
        if embeddings_file.endswith('.pkl') or embeddings_file.endswith('.pickle'):
            with open(embeddings_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
        elif embeddings_file.endswith('.npy'):
            # If it's a numpy array, assume it's in the same order as self.texts
            embeddings_array = np.load(embeddings_file)
            if len(embeddings_array) != len(self.texts):
                raise ValueError(f"Embeddings array length ({len(embeddings_array)}) doesn't match texts length ({len(self.texts)})")
            return embeddings_array
        else:
            raise ValueError("Unsupported file format. Use .pkl, .pickle, or .npy")
        
        # Match texts to embeddings
        embeddings = []
        not_found = []
        
        for text in self.texts:
            if text in embeddings_dict:
                embeddings.append(embeddings_dict[text])
            else:
                not_found.append(text[:50] + "..." if len(text) > 50 else text)
        
        if not_found:
            self._print(f"Warning: {len(not_found)} texts not found in embeddings file")
            if len(not_found) <= 5:
                self._print("Missing texts:", not_found)
            raise ValueError(f"{len(not_found)} texts not found in embeddings file")
        
        return np.array(embeddings)
    
    def _save_embeddings_to_file(self, save_path):
        """Save embeddings to file for future use."""
        self._print(f"Saving embeddings to: {save_path}")
        
        if save_path.endswith('.pkl') or save_path.endswith('.pickle'):
            # Save as dictionary mapping texts to embeddings
            embeddings_dict = {text: embedding for text, embedding in zip(self.texts, self.embeddings)}
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
        elif save_path.endswith('.npy'):
            # Save as numpy array
            np.save(save_path, self.embeddings)
        else:
            raise ValueError("Unsupported save format. Use .pkl, .pickle, or .npy")
        
        self._print("Embeddings saved successfully")

    def reduce(self, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 1):
        """
        Reduce the dimensionality of embeddings using UMAP.
        
        Parameters:
        -----------
        n_neighbors : int, optional (default=15)
            Number of neighbors to consider for each point in UMAP.
            Lower values emphasize local structure, higher values global structure.
        seed : int, optional (default=1)
            Random seed for reproducibility.
            
        Note:
        -----
        The reduced embeddings are stored in self.reduced_embeddings for
        visualization and further analysis.
        """
        # Convert embeddings to numpy array if not already
        embeddings_array = np.array(self.embeddings)
        
        # Create and configure UMAP reducer
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,  # Number of neighbors for local structure
            n_components=2,           # Reduce to 2D for visualization
            min_dist = min_dist,
            random_state=seed,           # For reproducibility
            n_jobs=1                  # Number of parallel jobs
        )
   
        # Perform dimensionality reduction
        self.reduced_embeddings = reducer.fit_transform(embeddings_array)
        
        self._print(f"Reduced embeddings to shape: {self.reduced_embeddings.shape}")

        return self
    
    def cluster(self, min_cluster_size: int = 5, min_samples: int = None, cluster_selection_epsilon: float = 0.0):
        """
        Perform clustering using HDBSCAN and save its condensed tree and labels.
        Also instantiate a helper object for hierarchical exploration.
        """
        data_array = np.array(self.reduced_embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        clusterer.fit(data_array)

        self.clusterer = clusterer

        self._print("Clustering complete.")
        n_clusters = len(set(self.clusterer.labels_)) - (1 if -1 in self.clusterer.labels_ else 0)
        self._print(f"Identified leaf {n_clusters} clusters.")

        self._compute_tree()

        return self
    
    def _compute_tree(self):
        df = self.clusterer.condensed_tree_.to_pandas()
        
        self.clusters = {}
        
        for i, row in df.iterrows():
            parent = int(row['parent'])
            if parent not in self.clusters:
                self.clusters[parent] = {"id": parent, "parent": -1,  "children": [], "points": [], "level": 0}
        
            child = int(row['child'])
        
            if row['child_size'] == 1:
                self.clusters[parent]['points'].append(child)
            else:
                if child not in self.clusters:
                    self.clusters[child] = {"id": child, "parent": -1,  "children": [], "points": [], "level": row['lambda_val']}
            
                self.clusters[parent]['children'].append(child)
                self.clusters[child]['parent'] = parent

        ls = [self.clusters[x]['level'] for x in self.clusters]
        ls = sorted(list(set(ls)))
        ls_map = {l: len(ls) - i - 1 for i, l in enumerate(ls)}
        
        self.max_level = max(ls_map.values())

        for x in self.clusters:
            self.clusters[x]['level'] = ls_map[self.clusters[x]['level']]
            
        root = [cl for cl in self.clusters if self.clusters[cl]['parent'] == -1]
        
        if len(root) > 1:
            raise ValueError("Assumption of a single root in a tree is violated")
        root = root[0]
        
        self._get_points(root)

    def _get_points(self, idn):
        cl = self.clusters[idn]
    
        if cl['children']:
            total = []
            for child in cl['children']:
                total += self._get_points(child)
            cl['points'] += total
          
        return cl['points']    
    
    def _get_level_clusters(self, level: int = 0):
        """
        Get clusters at a specific level of aggregation.
        
        Args:
            level: Level of aggregation to retrieve
        """

        level_clusters = []

        for cl_id in self.clusters:
            cl = self.clusters[cl_id]

            # We don't include clusters that are below the level threshold
            if cl['level'] < level: continue

            # We also need to skip clusters with children above or at the lelvel threshold

            if cl['children']:
                child = cl['children'][0]
                if self.clusters[child]['level'] >= level: continue

            level_clusters.append(cl)  

        return level_clusters  
    
    def get_cluster_labels(self, level: int = 0):
        """
        Get cluster labels for all points at a specific level of aggregation.
        
        Args:
            level: Level of aggregation to retrieve cluster labels for
            
        Returns:
            np.array: Array of cluster labels, one per data point. 
                    Points not in any cluster are labeled as -1 (noise).
        """
        # Initialize all points as noise (-1)
        labels = np.full(len(self.titles), -1, dtype=int)
        
        # Get clusters at the specified level
        level_clusters = self._get_level_clusters(level)
        
        # Assign cluster labels to points
        for cluster in level_clusters:
            cluster_id = cluster['id']
            for point_idx in cluster['points']:
                labels[point_idx] = cluster_id
        
        return labels

    def print_clusters(self, level: int = 0):
        clusters_to_print = self._get_level_clusters(level)

        for cluster in clusters_to_print:
            print('=' * 3, 'Cluster', cluster['id'], f'({len(cluster["points"])} items)', '=' * 3)
        
            if 'annotation' in cluster:
                print(cluster['annotation'])
            print()
            
            # Display up to 5 titles
            titles_to_show = min(5, len(cluster['points']))
            for i in range(titles_to_show):
                print(self.titles[cluster['points'][i]])
            
            # Add ellipsis if there are more titles
            if len(cluster['points']) > 5:
                print('...')
            
            print()
            print()

    def plot(self, marker_size: float = 5, alpha: float = 1, ax = None, groups = None, cluster_level = None, line_width: float = 0.5):
        """
        Plot reduced dimensions as a minimalistic scatter plot.
        
        Args:
            marker_size: Size of markers in scatter plot
            alpha: Transparency of markers (only applies to black/group 0 points)
            ax: Optional matplotlib axis to plot on. If None, creates new figure.
            groups: List of group indices for coloring points
        """
        # Set up the figure or use provided axis
        if ax is None:
            plt.figure(figsize=(6, 6))
            ax = plt.gca()

        blue = ["#264653", "#023e8a", "#0077b6"]
        green = ["#287271", "#2a9d8f", "#8ab17d"]
        yellow = ["#e9c46a", "#cda052"]
        orange = ["#e76f51", "#ee8959", "#f4a261"]
        red = ["#941c2f", "#c05761"]
        violet= ["#734f5a"]
        grey = ["#6c788b"]
        brown = ["#5a1c0c"]
        palette = ['#000000', blue[0], orange[0], green[0], red[0], yellow[0], violet[0], grey[0], brown[0], blue[1], orange[1], green[1], red[1], yellow[1], blue[2], orange[2], green[2]]
        if groups is None:
            groups = [0] * len(self.titles)

        colors = [palette[i % len(palette)] for i in groups]
        
        # Create alpha values: alpha for black points (group 0), 1.0 for others
        alpha_values = [alpha if group == 0 else 1.0 for group in groups]
        
        # Create scatter plot
        ax.scatter(
            self.reduced_embeddings[:, 0],
            self.reduced_embeddings[:, 1],
            s=marker_size,
            alpha=alpha_values,
            color=colors,
            edgecolors='none'  # This removes the marker border
        )
        
        if cluster_level is not None:
            self._plot_clusters(ax, cluster_level, line_width)

        # Set tight limits to eliminate white borders
        x_min, x_max = self.reduced_embeddings[:, 0].min(), self.reduced_embeddings[:, 0].max()
        y_min, y_max = self.reduced_embeddings[:, 1].min(), self.reduced_embeddings[:, 1].max()

        if cluster_level is not None:
            border_factor = 0.05
        else:
            border_factor = 0.005

        x_border = (x_max - x_min) * border_factor
        y_border = (y_max - y_min) * border_factor
        ax.set_xlim(x_min - x_border, x_max + x_border)
        ax.set_ylim(y_min - y_border, y_max + y_border)
        
        # Remove all decorations
        ax.axis('off')
        
        # Only call tight_layout if we created our own figure
        if ax == plt.gca() and plt.get_fignums():
            plt.tight_layout()

        return ax
    
    def _plot_clusters(self, ax, level, line_width: float = 0.5):
        clusters_to_display = self._get_level_clusters(level)

        for i, cluster in enumerate(clusters_to_display):
            if len(cluster['points']) > 2:  # Only draw contours for clusters with enough points
                # Get points for this cluster
                cluster_points = self.reduced_embeddings[cluster['points']]
                
                # Draw contour using KDE plot
                sns.kdeplot(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    levels=[0.1],  # Single contour level
                    color='black',
                    linestyles='--',
                    alpha=1,
                    zorder=-1,
                    linewidths=line_width,
                    ax=ax
                )

    def annotate(self, method='hf', model_name=None, system_prompt=None, 
                 max_titles_per_cluster=10, level=None, custom_function=None, device=None):
        """
        Generate annotations for clusters using various LLM providers.
        
        Args:
            method (str): Annotation method - 'hf', 'openai', 'gemini', or 'custom'
                         model_name (str): Model name for the LLM provider. Defaults:
                 - 'hf': 'Qwen/Qwen2.5-7B-Instruct'
                 - 'openai': 'gpt-4.1-nano'
                 - 'gemini': 'gemini-1.5-flash'
                         system_prompt (str): Custom system prompt. If None, uses default prompt.
             max_titles_per_cluster (int): Maximum number of titles to send to LLM per cluster
             level (int): Cluster level to annotate. If None, annotates all levels (default: None)
             custom_function (callable): Custom annotation function that takes titles and returns annotation
             device (str): Device to use for HF models ('mps', 'cuda', 'cpu'). If None, auto-detects best available.
        
        Returns:
            self: Returns self for method chaining
            
        Examples:
            # Use built-in methods
            latte.annotate(method='hf', model_name='Qwen/Qwen2.5-7B-Instruct')
            latte.annotate(method='hf', device='mps')  # Force MPS on Mac
            latte.annotate(method='openai', model_name='gpt-4.1-nano')
            latte.annotate(method='gemini')
            
            # Use custom function
            def my_annotation_function(titles):
                # Your custom annotation logic here
                return "Custom annotation"
            
            latte.annotate(method='custom', custom_function=my_annotation_function)
            
            # Or simply pass the function directly
            latte.annotate(custom_function=my_annotation_function)
        """
        if not hasattr(self, 'clusters') or not self.clusters:
            raise ValueError("No clusters found. Please run cluster() method first.")
        
        # Set default model names if not provided
        if model_name is None:
            if method == 'hf':
                model_name = 'Qwen/Qwen2.5-7B-Instruct'
            elif method == 'openai':
                model_name = 'gpt-4.1-nano'
            elif method == 'gemini':
                model_name = 'gemini-1.5-flash'
        
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that identifies the main topic or key topics from a set of texts. You must provide only the main topics in a single, concise sentence. Do not add any introductions or explanations. Do not start with phrases like 'The texts cover', 'The texts focus on', 'These texts are about', or similar introductory phrases. Just state the topic directly."
        
        # Get clusters at the specified level or all clusters if level is None
        if level is None:
            # Get all clusters that have points (leaf clusters)
            clusters_to_annotate = [cluster for cluster in self.clusters.values() if cluster['points']]
            self._print(f"Annotating {len(clusters_to_annotate)} clusters across all levels...")
        else:
            clusters_to_annotate = self._get_level_clusters(level)
            self._print(f"Annotating {len(clusters_to_annotate)} clusters at level {level}...")
        
        # If custom_function is provided, use it regardless of method
        if custom_function is not None:
            if not callable(custom_function):
                raise ValueError("custom_function must be a callable that takes a list of titles and returns an annotation string")
            annotation_function = custom_function
        elif method == 'custom':
            if custom_function is None:
                raise ValueError("custom_function must be provided when method='custom'")
        elif method == 'hf':
            annotation_function = lambda titles: annotate_with_hf(titles, model_name, system_prompt, device)
        elif method == 'openai':
            annotation_function = lambda titles: annotate_with_openai(titles, model_name, system_prompt)
        elif method == 'gemini':
            annotation_function = lambda titles: annotate_with_gemini(titles, model_name, system_prompt)
        else:
            raise ValueError(f"Unknown annotation method: {method}. Available methods: 'hf', 'openai', 'gemini', 'custom'")
        
        # Annotate each cluster
        for i, cluster in enumerate(clusters_to_annotate):
            cluster_id = cluster['id']
            cluster_points = cluster['points']
            
            if len(cluster_points) == 0:
                self._print(f"Skipping empty cluster {cluster_id}")
                continue
            
            # Randomly select up to max_titles_per_cluster titles
            if len(cluster_points) > max_titles_per_cluster:
                selected_indices = random.sample(cluster_points, max_titles_per_cluster)
            else:
                selected_indices = cluster_points
            
            # Get the titles for selected points
            selected_titles = [self.titles[idx] for idx in selected_indices]
            
            try:
                self._print(f"Annotating cluster {cluster_id} ({i+1}/{len(clusters_to_annotate)}) with {len(selected_titles)} titles...")
                
                # Generate annotation
                annotation = annotation_function(selected_titles)
                
                # Clean up the annotation
                annotation = self._clean_annotation(annotation)
                
                # Store annotation in cluster
                self.clusters[cluster_id]['annotation'] = annotation
                
                self._print(f"Cluster {cluster_id} annotation: {annotation}")
                
            except Exception as e:
                self._print(f"Error annotating cluster {cluster_id}: {str(e)}")
                self.clusters[cluster_id]['annotation'] = f"Error: {str(e)}"
        
        self._print("Annotation complete!")
        return self

    def _clean_annotation(self, annotation):
        """
        Clean up annotation text by removing unwanted punctuation and formatting.
        
        Args:
            annotation (str): Raw annotation from LLM
            
        Returns:
            str: Cleaned annotation
        """
        if not annotation:
            return annotation
        
        # Strip whitespace
        cleaned = annotation.strip()
        
        # Remove trailing punctuation (dots, commas, semicolons, etc.)
        while cleaned and cleaned[-1] in '.,:;!?':
            cleaned = cleaned[:-1].strip()
        
        # Remove quotes if they wrap the entire annotation
        if len(cleaned) >= 2:
            if (cleaned.startswith('"') and cleaned.endswith('"')) or \
               (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1].strip()
        
        # Ensure first letter is capitalized
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned

    def export(self, filename, mask=None):
        """
        Export complete cluster analysis structure to JSON format.
        
        Args:
            filename (str): Path to save the JSON file
            mask (array-like, optional): Binary mask of the same length as titles.
                                       If provided, computes proportion and normalized proportion
                                       of points with value 1 for each cluster.
        
        Returns:
            self: Returns self for method chaining
        """
        import json
        
        if not hasattr(self, 'clusters') or not self.clusters:
            raise ValueError("No clusters found. Please run cluster() method first.")
        
        # Validate mask if provided
        if mask is not None:
            mask = np.array(mask)
            if len(mask) != len(self.titles):
                raise ValueError(f"Mask length ({len(mask)}) must match titles length ({len(self.titles)})")
            if not np.all(np.isin(mask, [0, 1])):
                raise ValueError("Mask must contain only binary values (0 and 1)")
        
        # Prepare export data with optimized structure
        export_data = {
            'metadata': {
                'total_documents': len(self.titles),
                'total_clusters': len(self.clusters),
                'max_level': getattr(self, 'max_level', 0)
            },
            'titles': self.titles,  # Save all titles once at the top level
            'clusters': []
        }
        
        # Add mask metadata if provided
        if mask is not None:
            f_base = float(np.mean(mask))
            export_data['metadata']['mask_baseline_proportion'] = f_base
            
            # Compute proportions for all clusters first to find min/max
            cluster_proportions = {}
            for cluster_id, cluster in self.clusters.items():
                if len(cluster['points']) > 0:
                    cluster_mask_values = mask[cluster['points']]
                    proportion = float(np.mean(cluster_mask_values))
                    cluster_proportions[cluster_id] = proportion
                else:
                    cluster_proportions[cluster_id] = 0.0
            
            # Find min and max proportions across all clusters
            f_values = list(cluster_proportions.values())
            f_min = min(f_values)
            f_max = max(f_values)
            
            export_data['metadata']['mask_min_proportion'] = f_min
            export_data['metadata']['mask_max_proportion'] = f_max
        
        # Export all clusters in the hierarchical structure
        for cluster_id, cluster in self.clusters.items():
            cluster_data = {
                'id': cluster['id'],
                'level': cluster['level'],
                'size': len(cluster['points']),
                'point_indices': cluster['points'],  # Save indices instead of full titles
                'parent_id': cluster['parent'] if cluster['parent'] != -1 else None,
                'children_ids': cluster['children'],
                'annotation': cluster.get('annotation', None)
            }
            
            # Add mask-based metrics if mask is provided
            if mask is not None:
                f = cluster_proportions[cluster_id]
                cluster_data['mask_proportion'] = f
                
                # Compute normalized proportion
                if f == f_base:
                    normalized_proportion = 0.0
                elif f > f_base:
                    if f_max == f_base:
                        normalized_proportion = 0.0  # Handle edge case where all clusters have same proportion as baseline
                    else:
                        normalized_proportion = (f - f_base) / (f_max - f_base)
                else:  # f < f_base
                    if f_base == f_min:
                        normalized_proportion = 0.0  # Handle edge case where all clusters have same proportion as baseline
                    else:
                        normalized_proportion = (f - f_base) / (f_base - f_min)
                
                cluster_data['mask_normalized_proportion'] = float(normalized_proportion)
            
            export_data['clusters'].append(cluster_data)
        
        # Save to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self._print(f"Complete cluster structure exported to {filename}")
        return self

    def clear_annotation_cache(self):
        """
        Clear the cached annotation models to free up memory.
        
        This is useful when you want to free up GPU/CPU memory after annotation
        or when switching between different models.
        
        Returns:
            self: Returns self for method chaining
        """
        clear_model_cache()
        return self