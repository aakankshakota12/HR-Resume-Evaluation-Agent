from jd_profiling.query_interpreter import interpret_query
from jd_profiling.filter_engine import apply_filters

# ---- Mock Candidate Data ----
candidates = [
    {"name": "A", "skills": ["Python", "ML"], "score": 9, "experience": "internship"},
    {"name": "B", "skills": ["Java", "AWS"], "score": 7, "experience": "fresher"},
    {"name": "C", "skills": ["Python", "FastAPI", "AWS"], "score": 8, "experience": "internship"},
    {"name": "D", "skills": ["ML", "AI"], "score": 6, "experience": "fresher"},
]

queries = [
    "Top 2 candidates with Python ML",
    "Top 3 candidates with Python",
    "Best ML candidate",
    "Top candidates with Python and FastAPI"
]

for query in queries:
    print("\n==============================")
    print("Query:", query)

    intent = interpret_query(query)
    print("Parsed Intent:", intent)

    results = apply_filters(candidates, intent)

    print("Results:")
    for r in results:
        print(r)
