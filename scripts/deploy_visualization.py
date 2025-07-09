#!/usr/bin/env python3
"""
LATTE Visualization Deployment Script

This script creates a standalone web deployment package from a LATTE JSON export.
The resulting package can be uploaded to any web server for sharing visualizations.

Usage:
    python scripts/deploy_visualization.py data/results.json
    python scripts/deploy_visualization.py data/results.json --name my_analysis
    python scripts/deploy_visualization.py data/results.json --inline --zip
"""

import json
import shutil
import os
import argparse
import zipfile
from pathlib import Path

def read_json_file(json_path):
    """Read and validate JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic validation
        required_keys = ['metadata', 'titles', 'clusters']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Invalid LATTE JSON: missing '{key}' field")
        
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {json_path}")

def copy_viz_files(deploy_dir):
    """Copy all visualization files to deploy directory."""
    viz_dir = Path("viz")
    
    if not viz_dir.exists():
        raise ValueError("viz directory not found. Run this script from the project root.")
    
    # Create deploy directory
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files and subdirectories
    for item in viz_dir.iterdir():
        if item.name == 'embed_data.py':
            continue  # Skip development script
        
        dest = deploy_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
    
    print(f"✅ Copied visualization files to {deploy_dir}")

def embed_data_in_html(deploy_dir, json_data):
    """Embed JSON data in HTML and remove file upload functionality."""
    html_file = deploy_dir / "viewer.html"
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Create embedded data script
    js_data = f"    <!-- Embedded visualization data -->\n    <script>\n        const EMBEDDED_DATA = {json.dumps(json_data, indent=12)};\n    </script>"
    
    # Remove file upload elements from HTML
    # Remove file input container
    html_content = html_content.replace(
        '''            <div class="file-input-container">
                <label for="jsonFile" class="file-input-label" title="Load LATTE JSON file">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14,2 14,8 20,8"></polyline>
                    </svg>
                </label>
                <input type="file" id="jsonFile" accept=".json" style="display: none;">
            </div>''', 
        ''
    )
    
    # Add embedded data before JS modules
    html_content = html_content.replace(
        '    <!-- Load all JavaScript modules -->',
        js_data + '\n\n    <!-- Load all JavaScript modules -->'
    )
    
    # Write modified HTML
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Embedded data in HTML and removed file upload interface")

def restore_auto_loading(deploy_dir):
    """Restore auto-loading functionality in main.js."""
    main_js = deploy_dir / "js" / "main.js"
    
    with open(main_js, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Add back the auto-loading code
    auto_load_code = '''        
        // Auto-load embedded data
        if (typeof EMBEDDED_DATA !== 'undefined') {
            this.controls.loadEmbeddedData(EMBEDDED_DATA);
        }'''
    
    js_content = js_content.replace(
        '        };',
        '        };' + auto_load_code
    )
    
    with open(main_js, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print("✅ Restored auto-loading functionality")

def inline_assets(deploy_dir):
    """Inline CSS and JS files into HTML for single-file deployment."""
    html_file = deploy_dir / "viewer.html"
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Inline CSS
    css_file = deploy_dir / "css" / "styles.css"
    with open(css_file, 'r', encoding='utf-8') as f:
        css_content = f.read()
    
    html_content = html_content.replace(
        '<link rel="stylesheet" href="css/styles.css">',
        f'<style>\n{css_content}\n    </style>'
    )
    
    # Inline JavaScript files in order
    js_files = [
        "js/utils.js",
        "js/data-manager.js", 
        "js/tree-renderer.js",
        "js/annotation-system.js",
        "js/controls.js",
        "js/sidebar.js",
        "js/sidebar-resize.js",
        "js/smart-tooltips.js",
        "js/main.js"
    ]
    
    inlined_js = "\n\n    <!-- Inlined JavaScript modules -->\n    <script>"
    for js_file in js_files:
        js_path = deploy_dir / js_file
        with open(js_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
        inlined_js += f"\n        // === {js_file} ===\n        {js_content}\n"
    inlined_js += "    </script>"
    
    # Remove all JS script tags and replace with inlined version
    import re
    js_section_pattern = r'    <!-- Load all JavaScript modules -->.*?</body>'
    html_content = re.sub(js_section_pattern, 
                         f'    {inlined_js.strip()}\n</body>', 
                         html_content, flags=re.DOTALL)
    
    # Write single-file HTML
    standalone_file = deploy_dir / "index.html"
    with open(standalone_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Created standalone index.html with inlined assets")
    return standalone_file

def create_zip(deploy_dir, zip_name):
    """Create a zip file of the deployment."""
    zip_path = deploy_dir.parent / f"{zip_name}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in deploy_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(deploy_dir)
                zipf.write(file_path, arcname)
    
    print(f"✅ Created deployment package: {zip_path}")
    return zip_path

def main():
    parser = argparse.ArgumentParser(
        description="Deploy LATTE visualization for web sharing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/deploy_visualization.py data/biology_results.json
    python scripts/deploy_visualization.py data/results.json --name my_analysis
    python scripts/deploy_visualization.py data/results.json --inline --zip
        """
    )
    
    parser.add_argument('json_file', help='Path to LATTE JSON export file')
    parser.add_argument('--name', default='latte_visualization', 
                       help='Name for deployment folder/package (default: latte_visualization)')
    parser.add_argument('--inline', action='store_true',
                       help='Create single-file HTML with inlined CSS/JS')
    parser.add_argument('--zip', action='store_true',
                       help='Create a zip package of the deployment')
    
    args = parser.parse_args()
    
    try:
        print(f"🚀 Deploying visualization from {args.json_file}")
        
        # Read and validate JSON data
        json_data = read_json_file(args.json_file)
        print(f"📊 Loaded data: {json_data['metadata']['total_documents']} documents, "
              f"{json_data['metadata']['total_clusters']} clusters")
        
        # Create deployment directory
        deploy_dir = Path("deployments") / args.name
        
        # Copy files
        copy_viz_files(deploy_dir)
        
        # Embed data and modify HTML
        embed_data_in_html(deploy_dir, json_data)
        restore_auto_loading(deploy_dir)
        
        # Inline assets if requested
        if args.inline:
            # Create standalone HTML file
            standalone_file = inline_assets(deploy_dir)
            
            # Create clean standalone directory
            standalone_dir = Path("deployments") / f"{args.name}_single"
            standalone_dir.mkdir(parents=True, exist_ok=True)
            
            # Move standalone file to clean directory
            final_file = standalone_dir / "index.html"
            shutil.move(standalone_file, final_file)
            
            # Remove the regular deployment directory since we only want standalone
            shutil.rmtree(deploy_dir)
            
            print(f"📄 Standalone file ready: {final_file}")
            print(f"🗂️ Single-file deployment: {standalone_dir}")
        
        # Create zip if requested
        if args.zip:
            zip_path = create_zip(deploy_dir, args.name)
            
            # Remove the directory after creating zip (we only want the zip)
            shutil.rmtree(deploy_dir)
            
            print(f"📦 Deployment package: {zip_path}")
            print("🗂️ Folder removed - only zip file remains")
        
        print(f"\n🎉 Deployment complete!")
        
        if args.inline:
            print(f"🌐 Upload index.html to any web server")
        elif args.zip:
            print(f"🌐 Upload and extract zip file on web server")
        else:
            print(f"📁 Deploy folder: {deploy_dir}")
            print(f"🌐 Upload entire folder contents to web server")
            print(f"🏠 Main file: viewer.html")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 