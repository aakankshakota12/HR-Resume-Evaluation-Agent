import re

def interpret_query(user_query):
    query = user_query.lower()

    intent = "top_k"
    k = None
    order = "desc"
    strict = False

    include_skills = []
    exclude_skills = []
    min_score = None
    experience = None

    # -------- Intent & Ranking Detection --------
    top_match = re.search(r'top\s+(\d+)', query)
    number_match = re.search(r'\b(\d+)\b', query)

    if top_match:
        intent = "top_k"
        k = int(top_match.group(1))
    elif "best" in query:
        intent = "best_candidate"
        k = 1
    elif "top" in query:
        intent = "top_k"
        k = 10
    elif number_match and "score" not in query and "above" not in query:
        # If a number exists but not score-related → treat as ranking
        intent = "top_k"
        k = int(number_match.group(1))

    # -------- Sorting --------
    if "lowest" in query or "ascending" in query:
        order = "asc"

    # -------- Score Filter --------
    score_match = re.search(r'above\s+(\d+)', query)
    plus_match = re.search(r'(\d+)\+', query)

    if score_match:
        min_score = int(score_match.group(1))
    elif plus_match:
        min_score = int(plus_match.group(1))

    # -------- Experience --------
    if "internship" in query:
        experience = "internship"
    elif "fresher" in query:
        experience = "fresher"

    # -------- Strict Mode --------
    if "only" in query or "must have" in query:
        strict = True

    # -------- Exclude Skills --------
    if "without" in query or "exclude" in query:
        parts = re.split("without|exclude", query)
        if len(parts) > 1:
            exclude_words = re.findall(r'\b[a-zA-Z]+\b', parts[1])
            exclude_skills = [word.capitalize() for word in exclude_words]

    # -------- Skill Normalization --------
    skill_map = {
        "python": "Python",
        "ml": "ML",
        "aws": "AWS",
        "docker": "Docker",
        "ai": "AI",
        "fastapi": "FastAPI",
        "java": "Java",
        "data science": "Data Science"
    }

    # Detect "data science" phrase first
    if "data science" in query:
        include_skills.append("Data Science")

    words = re.findall(r'\b[a-zA-Z]+\b', query)

    for word in words:
        if word in skill_map:
            skill = skill_map[word]
            if skill not in include_skills and skill not in exclude_skills:
                include_skills.append(skill)

    return {
        "intent": intent,
        "k": k,
        "filters": {
            "include_skills": include_skills,
            "exclude_skills": exclude_skills,
            "min_score": min_score,
            "experience": experience
        },
        "sort_by": "score",
        "order": order,
        "strict": strict
    }
