# LATTE ☕ Interactive Visualization

This directory contains the interactive web-based visualization component for LATTE (LLM-Assisted Topic/Thematic Analysis).

## Project Overview

The LATTE project consists of two main parts:
1. **Python Analysis Engine** (main directory) - Handles embedding, clustering, and LLM annotation
2. **Interactive Visualization** (this directory) - Browser-based visualization of cluster results

## Architecture (Recently Refactored)

The visualization has been refactored from a single monolithic HTML file into a clean modular structure:

### File Structure
```
viz/
├── viewer.html              # Main HTML file (simplified, modular)
├── css/
│   └── styles.css           # All CSS styles
├── js/
│   ├── main.js              # Application initialization
│   ├── data-manager.js      # Data loading and management
│   ├── tree-renderer.js     # Tree visualization logic
│   ├── annotation-system.js # Annotation display and management
│   ├── controls.js          # UI controls and state management
│   └── utils.js             # Utility functions
├── embed_data.py            # Development helper (unchanged)
├── test.json                # Test data (unchanged)
└── viewer-original.html     # Backup of original monolithic version
```

### Benefits of Refactored Architecture
- **Clean Separation of Concerns**: Each module has a specific responsibility
- **Improved Maintainability**: Code is organized and easy to navigate
- **Better Extensibility**: New features can be added without affecting existing code
- **Easier Testing**: Individual components can be tested independently
- **Same Functionality**: All original features preserved exactly

### Modules Overview

**DataManager** (`data-manager.js`)
- Handles file loading and embedded data
- Validates data format
- Manages current dataset state

**TreeRenderer** (`tree-renderer.js`)
- D3.js tree visualization
- Node and link rendering
- Zoom and pan functionality

**AnnotationSystem** (`annotation-system.js`)
- Tooltip management
- "Show all annotations" feature
- Force simulation for annotation positioning

**ControlsManager** (`controls.js`)
- UI state management
- Toggle event handling
- Metadata display

**Utils** (`utils.js`)
- Color calculation utilities
- Text truncation helpers
- Math utilities

## Features

### Core Functionality ✅
- **Tree Structure Display**: Hierarchical visualization of clusters using D3.js tree layout
- **Hover Annotations**: Clean tooltips showing LLM-generated cluster annotations
- **Mask Proportion Coloring**: Color-coded nodes based on normalized proportion values
- **Show All Annotations**: Toggle to display all annotations simultaneously with collision avoidance
- **File Loading**: JSON file input for LATTE export data
- **Interactive Navigation**: Zoom and pan functionality

### Technical Implementation
- Built with D3.js v7 for visualization
- Vanilla JavaScript (no build process required)
- Force simulation for annotation positioning
- Layered circle approach for clean masking
- JSON data format from LATTE's `export()` method

## Data Format

The visualization expects JSON files exported from LATTE using the `export()` method:

```python
# In Python LATTE
latte.export('results.json')
```

Key data structure:
- `metadata`: Dataset statistics (documents, clusters, levels)
- `titles`: Array of document titles
- `clusters`: Array of cluster objects with hierarchy information and annotations

## Usage

### For Testing (with embedded data):
1. Open `viewer.html` in a web browser
2. Data loads automatically from embedded test data
3. All features work immediately

### For Production (with file upload):
1. Open `viewer.html` in a web browser
2. Click "Choose File" and select a LATTE JSON export
3. Explore the tree:
   - Hover over nodes to see annotations
   - Toggle "Show mask proportions" to see color-coded nodes
   - Toggle "Show all annotations" to see all labels at once
   - Use mouse wheel to zoom
   - Click and drag to pan

### Development
The `embed_data.py` script can still be used to embed different test data:
```bash
python embed_data.py
```

## Technical Requirements

- Modern web browser with JavaScript enabled
- No server required - runs entirely client-side
- No build process needed - uses vanilla JavaScript
- Compatible with LATTE JSON exports from any dataset

## Deployment

The refactored version maintains the same simplicity:
- All files are static assets
- Can be hosted on any web server or opened locally
- Self-contained with no external dependencies (except D3.js CDN) 