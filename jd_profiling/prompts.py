JD_PROMPT = """
You are an expert HR assistant.

Analyze the following Job Description and extract:

1. Required skills
2. Preferred skills
3. Experience required
4. Education required
5. Tools / Technologies mentioned

Return ONLY valid JSON in this exact format:

{{
  "required_skills": [],
  "preferred_skills": [],
  "experience_required": "",
  "education_required": "",
  "tools_technologies": []
}}

Job Description:
{jd}
"""
