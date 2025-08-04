from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib
import os
import sys
from dotenv import load_dotenv

# === Path Setup for gemini_handler ===
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
sys.path.append(utils_path)

from gemini_handler import get_gemini_response

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'
load_dotenv()

# === Load model and dataset ===
model = joblib.load("trained_model.pkl")
df = pd.read_csv("symptoms_disease_dataset.csv")

def get_details(disease):
    row = df[df['Disease'].str.lower() == disease.lower()].head(1)
    if not row.empty:
        return row.iloc[0]['Precaution'], row.iloc[0]['Medicine']
    return None, None

@app.route("/")
def index():
    session.clear()
    background_url = url_for('static', filename='img.avif')
    return render_template("welcom.html", background_url=background_url)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    background_url = url_for('static', filename='chatbot.avif')

    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        symptoms_input = request.form['symptoms'].lower().strip()
        symptoms_list = [s.strip() for s in symptoms_input.split(',')]

        # === Follow-up Step ===
        if len(symptoms_list) == 1 and not session.get('followup_done'):
            single_symptom = symptoms_list[0]
            symptom_exists = any(
                single_symptom in [s.strip().lower() for s in row['symptoms'].split(',')]
                for _, row in df.iterrows()
            )

            if symptom_exists:
                session['initial_symptom'] = symptoms_input
                session['followup_done'] = True
                output = (
                    f"<b>You entered only:</b> {symptoms_input}<br>"
                    "This symptom can relate to many diseases.<br>"
                    "⚠️ Please provide more symptoms (e.g., headache, fever, joint pain) for better accuracy."
                )
                session['chat_history'].append({
                    'user': symptoms_input,
                    'bot': output
                })
                session.modified = True
                return render_template("index.html", background_url=background_url)

        # === Append Follow-up input ===
        if session.get('followup_done'):
            first = session.pop('initial_symptom', '')
            session.pop('followup_done', None)
            symptoms_input = f"{first}, {symptoms_input}"
            symptoms_list = [s.strip() for s in symptoms_input.split(',')]

        # === Match from Dataset ===
        matched_row = None
        for _, row in df.iterrows():
            dataset_symptoms = [s.strip().lower() for s in row['symptoms'].split(',')]
            if all(symptom in dataset_symptoms for symptom in symptoms_list):
                matched_row = row
                break

        # === Output from Model/CSV ===
        if matched_row is not None:
            output = (
                f"<b>🦠 Disease:</b> {matched_row['disease']}<br>"
                f"<b>🔍 Symptoms:</b> {matched_row['symptoms']}<br>"
                f"<b>💊 Medicine:</b> {matched_row['medicine']}<br>"
                f"<b>🛡️ Precautions:</b> {matched_row['precautions']}"
            )
        else:
            # === Gemini Fallback ===
            gemini_reply = get_gemini_response(symptoms_input)
            output = gemini_reply

        # === Store to Chat History ===
        session['chat_history'].append({
            'user': symptoms_input,
            'bot': output
        })
        session.modified = True

    return render_template("index.html", background_url=background_url)

if __name__ == '__main__':
    print("✅ Smart Health Bot Running at http://127.0.0.1:5000")
    app.run(debug=True)
