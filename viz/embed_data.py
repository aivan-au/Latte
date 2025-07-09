#!/usr/bin/env python3
"""
Script to embed test.json data directly into viewer.html for easier testing.
This creates viewer_with_data.html that doesn't require file upload.
"""

import json

def embed_data():
    # Read the JSON data
    with open('test.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Read the HTML file
    with open('viewer.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Convert JSON to JavaScript variable with proper formatting
    js_data = f"const EMBEDDED_DATA = {json.dumps(json_data, indent=12)};"
    
    # Find the start and end of the existing EMBEDDED_DATA
    start_marker = "const EMBEDDED_DATA = {"
    end_marker = "};"
    
    start_pos = html_content.find(start_marker)
    if start_pos == -1:
        print("❌ Could not find existing EMBEDDED_DATA in viewer.html")
        print("💡 The new modular viewer.html should already have embedded data")
        return
    
    # Find the end position by counting braces
    brace_count = 0
    pos = start_pos + len(start_marker) - 1  # Position at the opening brace
    end_pos = -1
    
    for i in range(pos, len(html_content)):
        if html_content[i] == '{':
            brace_count += 1
        elif html_content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Check if the next non-whitespace character is ';'
                for j in range(i + 1, len(html_content)):
                    if html_content[j] == ';':
                        end_pos = j + 1
                        break
                    elif not html_content[j].isspace():
                        break
                break
    
    if end_pos == -1:
        print("❌ Could not find the end of EMBEDDED_DATA")
        return
    
    # Replace the data
    modified_html = html_content[:start_pos] + js_data + html_content[end_pos:]
    
    # Write the modified HTML to a new file
    with open('viewer_with_data.html', 'w', encoding='utf-8') as f:
        f.write(modified_html)
    
    print("✅ Created viewer_with_data.html with updated test data")
    print("📝 You can now open viewer_with_data.html and the data will load automatically")
    print("🔄 Run this script again whenever you update test.json")

if __name__ == "__main__":
    embed_data() 