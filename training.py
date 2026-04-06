import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -------- 1. LOAD DATASET --------
# sep=None with engine='python' tells pandas to guess if it's ; or ,
try:
    df = pd.read_csv("student-mat.csv", sep=None, engine='python')
    print(f"✅ Loaded file. Columns found: {df.columns.tolist()[:5]}...")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# -------- 2. TARGET & FEATURES --------
if "G3" not in df.columns:
    print("❌ Critical Error: 'G3' column not found. Check your CSV file.")
    exit()

df["pass"] = (df["G3"] >= 10).astype(int)

features = [
    "G1", "G2", "absences", "failures", "studytime", 
    "Medu", "Fedu", "goout", "health", "higher", "sex", "school"
]
X = df[features]
y = df["pass"]

# -------- 3. PIPELINE --------
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    ))
])

# -------- 4. TRAIN & SAVE --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"✅ Model Trained. Accuracy: {accuracy_score(y_test, y_pred):.2%}")

joblib.dump(pipeline, "model.joblib")
print("✅ Saved as model.joblib")
