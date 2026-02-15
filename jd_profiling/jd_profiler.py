import os
import json
import re
from google import genai
from dotenv import load_dotenv
from jd_profiling.prompts import JD_PROMPT

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def clean_json_text(text):
    """Helper to strip markdown code blocks."""
    text = text.strip()
    # Remove json and  if they exist
    if text.startswith(""):
        lines = text.splitlines()
        # Filter out the marker lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text

def profile_jd(jd_text):
    prompt = JD_PROMPT.format(jd=jd_text)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        cleaned_text = clean_json_text(response.text)
        
        # Locate the JSON object using regex as a backup
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response")

    except Exception as e:
        print(f"Error profiling JD: {e}")
        return {}