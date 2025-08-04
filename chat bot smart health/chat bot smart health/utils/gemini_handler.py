import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

def clean_gemini_output(raw_text):
    """
    Clean and summarize the Gemini response to include only
    Disease, Symptoms, Medicine, and Precautions in clean HTML format.
    """
    # Remove markdown artifacts
    raw_text = raw_text.replace("**", "").replace("*", "").replace("•", "-")

    disease = ""
    symptoms = ""
    medicine = ""
    precautions = ""
    description = ""

    for line in raw_text.splitlines():
        lower = line.lower()
        if "disease" in lower:
            disease += line.strip() + " "
        elif "symptom" in lower:
            symptoms += line.strip() + " "
        elif "medicine" in lower or "drug" in lower:
            medicine += line.strip() + " "
        elif "precaution" in lower or "care" in lower:
            precautions += line.strip() + " "
        elif len(line.strip()) > 15 and not any(keyword in lower for keyword in ["disease", "symptom", "medicine", "precaution"]):
            description += line.strip() + " "

    # Format for chatbot display
    formatted = ""
    if disease:
        formatted += f"<b>🦠 Disease:</b> {disease.strip()}<br>"
    if symptoms:
        formatted += f"<b>🔍 Symptoms:</b> {symptoms.strip()}<br>"
    if medicine:
        formatted += f"<b>💊 Recommended Medicine:</b> {medicine.strip()}<br>"
    if precautions:
        formatted += f"<b>🛡️ Precautions:</b> {precautions.strip()}<br>"
    if description:
        formatted += f"<b>📖 Description:</b> {description.strip()}<br>"

    return formatted.strip() if formatted else raw_text


def get_gemini_response(symptoms):
    prompt = (
        f"A user is experiencing these symptoms: {symptoms}.\n"
        "Please provide a summary with the following sections:\n"
        "1. Most likely disease\n"
        "2. Symptoms explained\n"
        "3. Recommended medicines or drugs\n"
        "4. Precautions or self-care tips\n"
        "5. A short description of the condition\n"
        "Avoid using ** or bullet points. Format it clearly for display in a chatbot interface."
    )

    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data)
        result = response.json()
        raw_text = result['candidates'][0]['content']['parts'][0]['text']
        return clean_gemini_output(raw_text)
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "⚠️ Gemini could not provide an answer right now. Please try again later."
