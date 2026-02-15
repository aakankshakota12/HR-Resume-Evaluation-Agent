JD_SCHEMA = {
    "required_skills": [],
    "preferred_skills": [],
    "experience_required": "",
    "education_required": "",
    "tools_technologies": []
}
QUERY_INTENT_SCHEMA = {
    "intent": "",          # top_k / best_candidate / filter
    "k": None,
    "filters": {
        "include_skills": [],
        "exclude_skills": [],
        "min_score": None,
        "experience": None
    },
    "sort_by": "score",
    "order": "desc",
    "strict": False
}


