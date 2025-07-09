import json
import csv
import os
import re
import html
from pathlib import Path
import py7zr

def clean_html(text):
    """Remove HTML tags and decode HTML entities from text"""
    if not text:
        return ""
    
    # First decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_tags(tags_string):
    """Extract tags from Stack Exchange tag format like '&lt;tag1&gt;&lt;tag2&gt;&lt;tag3&gt;'"""
    if not tags_string:
        return ""
    
    # First decode HTML entities to convert &lt; and &gt; to < and >
    decoded_tags = html.unescape(tags_string)
    
    # Extract tags using regex
    tags = re.findall(r'<([^>]+)>', decoded_tags)
    return "; ".join(tags)


def get_project_name(archive_name):
    """Extract simplified project name from archive name"""
    # Remove .stackexchange.com suffix
    project_name = archive_name.replace('.stackexchange.com', '')
    
    return project_name

def process_posts_file(posts_file_path, project_name):
    """Process a Posts.json file and extract questions"""
    print(f"Processing {posts_file_path} for project {project_name}")
    
    try:
        with open(posts_file_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)
    except Exception as e:
        print(f"Error loading {posts_file_path}: {e}")
        return []
    
    questions = []
    question_count = 0
    
    for post in posts:
        # Only process questions (PostTypeId = 1)
        if post.get('PostTypeId') == 1:
            question_count += 1
            
            # Extract required fields
            title = post.get('Title', '')
            body = clean_html(post.get('Body', ''))
            tags = extract_tags(post.get('Tags', ''))
            
            questions.append({
                'title': title,
                'text': body,
                'project': project_name,
                'tags': tags
            })
    
    print(f"Extracted {question_count} questions from {project_name}")
    return questions

def extract_and_process_7z(file_path, data_dir):
    """Extract 7z file and process Posts.json if it exists"""
    print(f"Extracting {file_path}")
    
    try:
        with py7zr.SevenZipFile(file_path, mode='r') as z:
            file_list = z.getnames()
            
            # Check if Posts.json exists in the archive
            if 'Posts.json' not in file_list:
                print(f"No Posts.json found in {file_path}")
                return []
            
            # Extract to a temporary directory named after the archive
            archive_name = Path(file_path).stem
            extract_dir = data_dir / f"temp_{archive_name}"
            extract_dir.mkdir(exist_ok=True)
            
            z.extractall(str(extract_dir))
            
            # Get simplified project name
            project_name = get_project_name(archive_name)
            
            # Process the Posts.json file
            posts_file = extract_dir / "Posts.json"
            questions = process_posts_file(posts_file, project_name)
            
            # Clean up extracted files
            import shutil
            shutil.rmtree(extract_dir)
            
            return questions, project_name
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], None

def find_data_dir():
    """Find the data directory relative to the current working directory"""
    # Try current directory first (if running from scripts/)
    if os.path.exists("../data"):
        return Path("../data")
    # Try data in current directory (if running from root)
    elif os.path.exists("data"):
        return Path("data")
    else:
        return None

def write_questions_to_csv(questions, project_name, data_dir):
    """Write questions to a CSV file named after the project"""
    if not questions:
        print(f"No questions to write for project {project_name}")
        return
    
    output_file = data_dir / f"{project_name}.csv"
    print(f"Writing {len(questions)} questions to {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'text', 'project', 'tags']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        writer.writeheader()
        for question in questions:
            writer.writerow(question)
    
    print(f"Successfully created {output_file} with {len(questions)} questions")

def main():
    """Main function to process all files in data directory"""
    data_dir = find_data_dir()
    
    if data_dir is None:
        print("Could not find 'data' directory. Please ensure it exists.")
        return
    
    total_questions = 0
    projects_processed = []
    
    # Process each 7z file separately
    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.name.endswith('.7z'):
            questions, project_name = extract_and_process_7z(file_path, data_dir)
            
            if questions and project_name:
                write_questions_to_csv(questions, project_name, data_dir)
                total_questions += len(questions)
                projects_processed.append((project_name, len(questions)))
    
    if not projects_processed:
        print("No questions found to process!")
        return
    
    print(f"\nSummary:")
    print(f"Total questions processed: {total_questions}")
    print(f"Projects processed:")
    for project, count in projects_processed:
        print(f"  {project}: {count} questions -> {project}.csv")

if __name__ == "__main__":
    main() 