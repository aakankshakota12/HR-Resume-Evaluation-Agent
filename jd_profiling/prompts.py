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
QUERY_PROMPT = """
You are an intelligent query interpreter for an AI-based resume ranking system.

Convert the user query into structured JSON.

The system already has:
- Ranked candidates with scores
- Extracted skills
- Experience information

Supported capabilities:

1. Top K candidates
2. Best candidate
3. Filter by skills
4. Filter by minimum score
5. Filter by experience
6. Combined filters
7. Sorting order (ascending / descending)

Return ONLY valid JSON in this exact format:

{{
  "intent": "",
  "k": null,
  "filters": {{
      "include_skills": [],
      "exclude_skills": [],
      "min_score": null,
      "experience": null
  }},
  "sort_by": "score",
  "order": "desc",
  "strict": false
}}

Rules:

- If user says "best", intent = "best_candidate"
- If user says "top N", intent = "top_k"
- If ranking requested but no number given, set k = 10
- If user says "lowest" or "ascending", set order = "asc"
- If user says "exclude" or "without", put skills in exclude_skills
- If user says "only" or "must have", set strict = true
- Extract ALL mentioned skills into include_skills
- Extract numeric score thresholds
- Do NOT include explanations
- The user may not explicitly mention "candidate".
- Any ranking-related phrase implies candidate ranking.
- Words like "top", "best", "highest", "lowest", "show", "list" imply ranking.
- If ranking is implied but no number is given, default k = 10.


User Query:
{query}
"""

