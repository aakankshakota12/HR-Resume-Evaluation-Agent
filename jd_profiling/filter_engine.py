def apply_filters(candidates, query_intent):
    filters = query_intent["filters"]
    include_skills = filters["include_skills"]
    exclude_skills = filters["exclude_skills"]
    min_score = filters["min_score"]
    experience = filters["experience"]
    strict = query_intent["strict"]

    order = query_intent["order"]
    k = query_intent["k"]

    filtered = []

    for candidate in candidates:
        candidate_skills = candidate.get("skills", [])
        candidate_score = candidate.get("score", 0)
        candidate_experience = candidate.get("experience", "")

        # ---- Exclude Skills ----
        if exclude_skills:
            if any(skill in candidate_skills for skill in exclude_skills):
                continue

        # ---- Include Skills ----
        skill_match_count = 0

        if include_skills:
            if strict:
                # Candidate must have ALL required skills
                if not all(skill in candidate_skills for skill in include_skills):
                    continue
                skill_match_count = len(include_skills)
            else:
                # Candidate must have at least ONE skill
                if not any(skill in candidate_skills for skill in include_skills):
                    continue
                skill_match_count = sum(
                    1 for skill in include_skills if skill in candidate_skills
                )

        # ---- Score Filter ----
        if min_score is not None:
            if candidate_score < min_score:
                continue

        # ---- Experience Filter ----
        if experience:
            if experience.lower() not in candidate_experience.lower():
                continue

        # ---- Smart Boost ----
        boosted_score = candidate_score + (skill_match_count * 0.2)

        candidate_copy = candidate.copy()
        candidate_copy["boosted_score"] = boosted_score

        filtered.append(candidate_copy)

    # ---- Sorting ----
    reverse_order = True if order == "desc" else False
    filtered.sort(
        key=lambda x: x.get("boosted_score", 0),
        reverse=reverse_order
    )

    # ---- Remove temporary boosted_score ----
    for candidate in filtered:
        candidate.pop("boosted_score", None)

    # ---- Top K ----
    if k is not None:
        filtered = filtered[:k]

    return filtered
