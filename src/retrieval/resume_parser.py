# src/retrieval/resume_parser.py
import os
import pdfplumber
import re
import hashlib

def clean_text(text):
    """
    Removes extra newlines, tabs, and multiple spaces to sanitize text.
    """
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_hash(file_path):
    """
    Generates a SHA256 hash of the file content to use as a unique ID.
    """
    sha = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return None

def load_resumes(resume_folder):
    """
    Iterates over PDFs, extracts text, and returns a list of dictionaries.
    """
    resumes_data = []
    
    if not os.path.exists(resume_folder):
        print(f"Error: Folder '{resume_folder}' not found.")
        return []

    print(f"Parsing resumes from: {resume_folder}...")

    for file in os.listdir(resume_folder):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(resume_folder, file)
            
            # Generate unique ID immediately
            file_hash = generate_hash(file_path)
            if not file_hash:
                continue

            try:
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            full_text += extracted + " "
                    
                    cleaned_text = clean_text(full_text)
                    
                    if cleaned_text:
                        resumes_data.append({
                            "id": file_hash,          # FIX: Use Hash instead of filename
                            "filename": file,         # Keep filename for reference
                            "text": cleaned_text,
                            "metadata": {"source": file_path}
                        })
                        print(f"  - Loaded: {file} (ID: {file_hash[:8]}...)")
                    else:
                        print(f"  - Warning: Empty text in {file}")

            except Exception as e:
                print(f"  - Error reading {file}: {e}")

    return resumes_data