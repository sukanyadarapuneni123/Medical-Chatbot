import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import requests
import json

# --------- MOCK DATASET ---------
data = {
    'symptoms': [
        ['fever', 'cough', 'headache'],
        ['sore throat', 'cough', 'fatigue'],
        ['chest pain', 'shortness of breath'],
        ['rash', 'itching', 'redness'],
        ['joint pain', 'swelling', 'stiffness'],
        ['fever', 'rash', 'body pain'],
        ['nausea', 'vomiting', 'stomach pain'],
        ['blurred vision', 'dizziness', 'headache']
    ],
    'disease': [
        'Flu',
        'Cold',
        'Heart Disease',
        'Allergy',
        'Arthritis',
        'Dengue',
        'Food Poisoning',
        'Migraine'
    ]
}

df = pd.DataFrame(data)

# Preprocess data
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['symptoms'])
y = df['disease']

# Train the ML model
model = DecisionTreeClassifier()
model.fit(X, y)

# --------- USER INPUT ---------
print("Enter symptoms separated by commas (e.g., fever,cough,headache):")
user_input_text = input("Symptoms: ").lower()
user_symptoms = [sym.strip() for sym in user_input_text.split(',')]

# Predict using ML
try:
    user_input_vector = mlb.transform([user_symptoms])
    ml_prediction = model.predict(user_input_vector)[0]
except Exception as e:
    ml_prediction = f"Unable to predict. Error: {str(e)}"

# --------- GEMINI API ---------
API_KEY = "AIzaSyDv_eIt-WOoA73Gqpvkdw1HcCTtHg1iW_k"  # Replace this with your actual Gemini API Key

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

prompt = f"User is experiencing the following symptoms: {', '.join(user_symptoms)}. What possible diseases could this indicate?"

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": prompt
                }
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=payload)

# Get Gemini API output
if response.status_code == 200:
    result = response.json()
    try:
        gemini_output = result['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError):
        gemini_output = "Unexpected format in Gemini response."
else:
    gemini_output = f"Gemini API error: {response.status_code} - {response.text}"

print(f"Predicted Disease: {ml_prediction}")


print(f"Possible Diseases: {gemini_output}")
