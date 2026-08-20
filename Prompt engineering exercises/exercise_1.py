import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

# Patient information
user_input = """
Patient ID: P1024
Age: 52
Gender: Male
Blood Group: O+
Chief Complaint: Chest pain for the last 2 hours
Symptoms: Chest pain, shortness of breath, sweating, mild dizziness
Onset: Sudden
Duration: 2 hours
Severity: 8/10
Current Diseases: Hypertension, Type 2 Diabetes
Current Medications: Metformin 500 mg twice daily, Amlodipine 5 mg once daily
Allergies: No known drug allergies
Past Medical History: Hypertension for 8 years, diabetes for 5 years
Past Surgical History: Appendectomy in 2015
Family History: Father had a history of coronary artery disease
Lifestyle: Smoker, approximately 5 cigarettes per day
Recent Tests: Blood pressure 160/95 mmHg, heart rate 102 bpm
Lab Results: Not available
Imaging Results: Not available
Additional Notes: Patient appears anxious and is experiencing persistent chest discomfort.
"""

output_format = """
Return ONLY valid JSON using the following structure:

{
    "patient_summary": "Concise doctor-friendly summary",
    "key_findings": [
        "Most important finding",
        "Second important finding"
    ],
    "symptoms": [
        "Symptom 1",
        "Symptom 2"
    ],
    "medical_history": [
        "Relevant disease or medical history"
    ],
    "medications": [
        "Medication name and dosage"
    ],
    "allergies": [
        "Allergy information"
    ],
    "surgical_history": [
        "Relevant previous surgery"
    ],
    "investigations": [
        "Important test or investigation result"
    ],
    "clinical_flags": [
        "Important information requiring medical attention"
    ]
}

Rules:
- Do not invent missing information.
- If information is unavailable, use an empty list [] or "Not available".
- Put the most clinically important information first.
- Keep the summary concise.
- Use professional and easy-to-understand medical language.
- Do not provide a diagnosis unless it is explicitly present in the patient information.
- Do not provide treatment recommendations.
"""

system_prompt = """
Role:
You are a medical patient-summary assistant.

Context:
You will receive structured or unstructured patient information. The information may include:
- Patient identification information
- Age and gender
- Chief complaint
- Presenting symptoms
- Symptom onset, duration, frequency and severity
- Vital signs
- Current medical conditions
- Past medical history
- Current medications and dosage
- Drug or food allergies
- Previous surgeries and procedures
- Family medical history
- Social and lifestyle history
- Recent hospitalizations or medical events
- Laboratory test results
- Imaging and diagnostic test results
- Physical examination findings
- Relevant clinical observations
- Previous diagnoses
- Relevant complications or risk factors
- Other clinically relevant notes

Task:
Create a concise patient summary using only the information provided.

Prioritization:
1. Chief complaint and most important symptoms
2. Important abnormal findings and vital signs
3. Relevant existing medical conditions
4. Current medications and allergies
5. Relevant medical and surgical history
6. Important investigation/test results
7. Other clinically relevant information

Constraints:
- Do not invent, assume, or infer information that is not provided.
- Do not omit clinically important information.
- Do not include unnecessary background information.
- Do not repeat the same information.
- Use concise, doctor-friendly language.
- Keep the most important information at the beginning.
- Do not make a diagnosis unless the diagnosis is explicitly provided.
- Do not recommend treatment or medication.
- If information is missing, clearly indicate that it is unavailable.
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not add explanations outside the JSON.

Output format:
""" + output_format


# Free model available on Groq
model = "openai/gpt-oss-20b"

# Generate summary
chat_completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_input
        }
    ],
    temperature=0.2,
    response_format={"type": "json_object"}
)

# Parse JSON returned by the model
result = json.loads(chat_completion.choices[0].message.content)

# Clean and concise display
print("\n" + "=" * 60)
print("PATIENT SUMMARY")
print("=" * 60)

print(f"\n{result['patient_summary']}")

print("\nKEY FINDINGS:")
for finding in result["key_findings"]:
    print(f"• {finding}")

print("\nSYMPTOMS:")
for symptom in result["symptoms"]:
    print(f"• {symptom}")

print("\nMEDICAL HISTORY:")
for history in result["medical_history"]:
    print(f"• {history}")

print("\nMEDICATIONS:")
for medication in result["medications"]:
    print(f"• {medication}")

print("\nALLERGIES:")
for allergy in result["allergies"]:
    print(f"• {allergy}")

print("\nSURGICAL HISTORY:")
for surgery in result["surgical_history"]:
    print(f"• {surgery}")

print("\nINVESTIGATIONS:")
for investigation in result["investigations"]:
    print(f"• {investigation}")

print("\nCLINICAL FLAGS:")
for flag in result["clinical_flags"]:
    print(f"• {flag}")

print("\n" + "=" * 60)