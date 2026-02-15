# src/retrieval/resume_parser.py
import os
import pdfplumber
import re

def clean_text(text):
    """
    Removes extra newlines, tabs, and multiple spaces to sanitize text
    for embeddings.
    """
    if not text:
        return ""
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
            try:
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            full_text += extracted + " "
                    
                    cleaned_text = clean_text(full_text)
                    
                    # Only add if text was actually found
                    if cleaned_text:
                        resumes_data.append({
                            "id": file,  # Using filename as ID
                            "text": cleaned_text,
                            "metadata": {"source": file_path}
                        })
                        print(f"  - Loaded: {file}")
                    else:
                        print(f"  - Warning: Empty text in {file}")

            except Exception as e:
                print(f"  - Error reading {file}: {e}")

    return resumes_data