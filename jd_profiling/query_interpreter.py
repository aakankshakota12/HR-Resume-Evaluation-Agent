import os
import json
import re
from google import genai
from dotenv import load_dotenv
from jd_profiling.prompts import QUERY_PROMPT
from jd_profiling.schemas import QUERY_INTENT_SCHEMA

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def clean_json_text(text):
    """Removes Markdown code blocks (```json ... ```) if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json) and last line (```)
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1])
    return text.strip()

def interpret_query(user_query):
    """
    Uses Gemini to translate natural language queries into structured filters.
    """
    prompt = QUERY_PROMPT.format(query=user_query)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        cleaned_text = clean_json_text(response.text)
        
        # Parse JSON
        intent_data = json.loads(cleaned_text)
        
        # Merge with default schema to ensure no missing keys
        final_intent = QUERY_INTENT_SCHEMA.copy()
        final_intent.update(intent_data)
        
        # Ensure 'filters' key exists and is merged correctly
        if "filters" in intent_data:
            default_filters = QUERY_INTENT_SCHEMA["filters"].copy()
            default_filters.update(intent_data["filters"])
            final_intent["filters"] = default_filters

        return final_intent

    except Exception as e:
        print(f"Error interpreting query: {e}")
        # Fallback: Return a default 'top 10' search if AI fails
        fallback = QUERY_INTENT_SCHEMA.copy()
        fallback["intent"] = "top_k"
        fallback["k"] = 10
        return fallback