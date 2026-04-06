import pandas as pd

# load dataset
df = pd.read_csv("student_performance_prediction.csv")

# -------- CLEAN --------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Columns:", df.columns)

# -------- FEATURE ENGINEERING --------

# 1. study efficiency
df["study_efficiency"] = df["hours"] * df["attendance"]

# 2. performance ratio
df["score_ratio"] = df["previous_score"] / 100

# 3. interaction feature
df["effort_score"] = df["hours"] * df["previous_score"]

# -------- TARGET --------
# ensure passes is 0/1
df["pass"] = df["pass"].astype(int)

# -------- SELECT FINAL FEATURES --------
df_final = df[[
    "hours",
    "attendance",
    "previous_score",
    "study_efficiency",
    "score_ratio",
    "effort_score",
    "pass"
]]

# save cleaned data
df_final.to_csv("cleaned_data.csv", index=False)

print("✅ Data preprocessing done")