from jd_profiling.jd_profiler import profile_jd

jd = """
We are hiring a Python Developer.
Must have: Python, FastAPI, SQL.
Preferred: AWS, Docker.
Experience: 2+ years.
Education: B.Tech in CSE or related field.
"""

result = profile_jd(jd)

print(result)
