import os
import json
import re
from google import genai
from dotenv import load_dotenv
from jd_profiling.prompts import JD_PROMPT

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def profile_jd(jd_text):
    prompt = JD_PROMPT.format(jd=jd_text)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    # Try parsing directly
    try:
        return json.loads(text)
    except:
        # Extract JSON safely
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            print("Model did not return valid JSON")
            print("Raw output:", text)
            return {}
