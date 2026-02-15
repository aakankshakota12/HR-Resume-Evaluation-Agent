def apply_filters(candidates, query_intent):
    filters = query_intent.get("filters", {})
    
    # Normalize filters for easier matching
    include_skills = [s.lower() for s in filters.get("include_skills", [])]
    exclude_skills = [s.lower() for s in filters.get("exclude_skills", [])]
    min_score = filters.get("min_score")
    req_experience = filters.get("experience")
    strict = query_intent.get("strict", False)

    order = query_intent.get("order", "desc")
    k = query_intent.get("k")

    filtered = []

    for candidate in candidates:
        # Normalize candidate data
        cand_skills = [s.lower() for s in candidate.get("skills", [])]
        cand_score = candidate.get("score", 0)
        cand_experience = candidate.get("experience", "").lower()

        # ---- 1. Exclude Skills ----
        if exclude_skills:
            if any(skill in cand_skills for skill in exclude_skills):
                continue

        # ---- 2. Include Skills ----
        skill_match_count = 0
        if include_skills:
            matches = [s for s in include_skills if s in cand_skills]
            skill_match_count = len(matches)
            
            if strict:
                # MUST have ALL skills
                if skill_match_count < len(include_skills):
                    continue
            else:
                # MUST have at least ONE skill (if skills were asked for)
                if skill_match_count == 0:
                    continue

        # ---- 3. Score Filter ----
        if min_score is not None:
            if cand_score < min_score:
                continue

        # ---- 4. Experience Filter (Improved) ----
        if req_experience:
            req_exp = req_experience.lower()
            # If user asks for "intern", we accept "internship", "intern", "trainee"
            # If user asks for "fresher", we accept "fresher", "junior", "entry"
            
            is_match = False
            if "intern" in req_exp and "intern" in cand_experience:
                is_match = True
            elif "fresh" in req_exp and ("fresh" in cand_experience or "entry" in cand_experience):
                is_match = True
            elif req_exp in cand_experience: # Fallback for other terms
                is_match = True
            
            if not is_match:
                continue

        # ---- 5. Smart Boost ----
        # Add 0.5 bonus for every matched skill to push relevant candidates up
        boosted_score = cand_score + (skill_match_count * 0.5)

        candidate_copy = candidate.copy()
        candidate_copy["boosted_score"] = boosted_score
        filtered.append(candidate_copy)

    # ---- Sorting ----
    reverse_order = True if order == "desc" else False
    
    # Sort by boosted score (relevance) instead of raw score
    filtered.sort(
        key=lambda x: x.get("boosted_score", 0),
        reverse=reverse_order
    )

    # Clean up temp field
    for candidate in filtered:
        candidate.pop("boosted_score", None)

    # ---- Top K ----
    if k is not None:
        filtered = filtered[:k]

    return filtered