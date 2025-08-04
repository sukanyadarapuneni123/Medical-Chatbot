import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Step 1: Load dataset
df = pd.read_csv('symptoms_disease_dataset.csv')  # Make sure this file is in the same directory

# Step 2: Split features and labels
X = df['symptoms']
y = df['disease']

# Step 3: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build pipeline with TF-IDF + Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Step 5: Train model
pipeline.fit(X_train, y_train)

# Step 6: Evaluate (optional but useful)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 7: Save the pipeline
joblib.dump(pipeline, 'trained_model.pkl')
print("Model saved as trained_model.pkl")
