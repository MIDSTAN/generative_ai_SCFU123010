import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# 1. Load API
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ---------------------------------------------------------
# 2. Job Profile
# ---------------------------------------------------------

job_profile = """
Job Role: Machine Learning Engineer

Job Description:
A Machine Learning Engineer is responsible for designing, developing,
training, evaluating, and deploying machine learning models. The role
requires strong knowledge of Python, machine learning algorithms,
deep learning, data preprocessing, model evaluation, and deployment.

Required Skills:
- Python programming
- NumPy, Pandas, Matplotlib
- Scikit-learn
- Machine Learning algorithms
- Deep Learning
- Neural Networks
- CNN, RNN, LSTM, Transformers
- PyTorch or TensorFlow
- Natural Language Processing
- Computer Vision
- SQL
- Data preprocessing and feature engineering
- Model evaluation and optimization
- Git and GitHub
- REST APIs
- Docker
- Basic cloud computing
- MLOps fundamentals
- Model deployment

Preferred Qualifications:
- Bachelor's degree in Computer Science or related field
- Hands-on machine learning projects
- Experience working with real-world datasets
- Knowledge of Generative AI and Large Language Models
- Understanding of model deployment and production workflows
- Good problem-solving and analytical skills
"""


# ---------------------------------------------------------
# 3. User Profile
# ---------------------------------------------------------

user_profile = """
Name: Pruthviraj Santosh Sapate

Education:
- B.Tech in Computer Science and Engineering
- Vishwa Prayag University, Solapur, Maharashtra

Technical Skills:
- Python
- C
- C++
- JavaScript
- React.js
- Django
- SQL
- MySQL
- MongoDB
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- TensorFlow
- CNN
- Vision Transformers
- Generative AI
- GANs
- Natural Language Processing
- Computer Vision
- Git and GitHub
- REST APIs

Machine Learning / AI Experience:
- Working on deepfake detection
- Researching cross-modal deepfake detection using image and video data
- Exploring GAN-based approaches for deepfake and defect detection
- Interested in using GAN discriminators as classifiers
- Working with CNN and Vision Transformer architectures
- Experience with prompt engineering and LLM APIs
- Experience using NVIDIA AI APIs and OpenAI-compatible APIs

Projects:

1. Cross-Modal Deepfake Detection
   - Detection of manipulated image and video content
   - CNN and Vision Transformer based architecture
   - Exploring GAN-based discriminator approaches
   - Focus on improving detection of generated and manipulated content

2. GAN-Based Defect Detection
   - Exploring the use of GAN generators to create difficult synthetic samples
   - Using discriminator feedback for classification and detection
   - Investigating adaptive adversarial training

3. Prompt Engineering
   - Building prompts for LLM-based applications
   - Working with structured outputs and API-based LLM systems
   - Experience with NVIDIA AI APIs

4. Full-Stack Applications
   - React.js frontend development
   - Django backend development
   - MySQL and MongoDB database integration
   - REST API development

Current Strengths:
- Python programming
- Machine Learning fundamentals
- Deep Learning
- Computer Vision
- GANs
- CNNs
- Vision Transformers
- Generative AI
- Full-stack development
- Problem solving

Areas to Improve:
- Advanced Machine Learning
- Advanced Deep Learning
- MLOps
- Docker
- Cloud platforms
- Model deployment
- Production ML systems
- Advanced mathematics for ML
- Distributed model training
- System design for ML applications
- Advanced SQL
- Real-world ML engineering practices

Career Goal:
Become a job-ready Machine Learning Engineer with strong practical skills
in machine learning, deep learning, computer vision, generative AI,
model deployment, and MLOps.
"""


# ---------------------------------------------------------
# 4. Two-Way Pipeline Prompt
# ---------------------------------------------------------

system_prompt = """
You are an expert Job-to-Candidate Analysis and Outreach Pipeline.

You have TWO responsibilities.

=========================================================
PIPELINE 1: HR OUTREACH
=========================================================

Compare the candidate profile with the job description.

Generate a professional and personalized email to the HR/recruiter.

The email should:
- Have a professional subject
- Introduce the candidate
- Mention the strongest matching skills
- Mention 1-2 highly relevant projects
- Explain why those projects are relevant to the job
- Mention the candidate's degree
- Show genuine interest in the role
- Request an opportunity to discuss the position
- NOT falsely claim experience that is not present
- Be concise and ready to send

=========================================================
PIPELINE 2: JOB GAP + LEARNING ANALYSIS
=========================================================

Analyze the job description against the candidate profile.

Identify:

1. Skills already matching the job
2. Skills partially matching the job
3. Missing skills
4. Skills that need improvement
5. Priority of each missing/improvable skill
6. What exactly the candidate should learn
7. Practical projects or tasks that would demonstrate that skill
8. Suggested learning order

For every missing or weak skill, explain it in simple language.

Priority should be one of:

- HIGH
- MEDIUM
- LOW

Focus especially on skills that are important for becoming
job-ready for THIS PARTICULAR job.

Do not recommend random technologies that are unrelated to the JD.

=========================================================
OUTPUT FORMAT
=========================================================

Return ONLY valid JSON.

Use this structure:

{
    "job_analysis": {
        "job_role": "",
        "job_summary": "",
        "match_percentage": 0,
        "strong_matches": [],
        "partial_matches": [],
        "missing_skills": []
    },

    "outreach": {
        "subject": "",
        "email": ""
    },

    "learning_plan": [
        {
            "skill": "",
            "current_status": "",
            "priority": "",
            "why_needed": "",
            "what_to_learn": [],
            "practical_task": ""
        }
    ],

    "recommended_order": []
}

Important:
- Do not invent experience.
- Use only information provided in the candidate profile.
- Match the learning recommendations specifically to the job.
- Keep the email concise.
- Keep explanations easy to understand.
"""


# ---------------------------------------------------------
# 5. User Prompt
# ---------------------------------------------------------

user_prompt = f"""
Analyze the following candidate and job description.

JOB PROFILE:
{job_profile}

CANDIDATE PROFILE:
{user_profile}

Generate both pipelines:

1. HR outreach email
2. Job gap analysis + learning plan

Return only valid JSON.
"""


# ---------------------------------------------------------
# 6. Call NVIDIA Model
# ---------------------------------------------------------

completion = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",

    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],

    temperature=0.3,
    top_p=0.9,
    max_tokens=12000,

    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "reasoning_budget": 8000
    },

    stream=True
)


# ---------------------------------------------------------
# 7. Collect Response
# ---------------------------------------------------------

response = ""

for chunk in completion:

    if not chunk.choices:
        continue

    content = chunk.choices[0].delta.content

    if content is not None:
        response += content


# ---------------------------------------------------------
# 8. Convert JSON Response
# ---------------------------------------------------------

try:

    # Remove possible markdown JSON wrapper
    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]

    if response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    result = json.loads(response)

except json.JSONDecodeError:

    print("Model did not return valid JSON.")
    print(response)
    exit()


# ---------------------------------------------------------
# 9. PIPELINE 1 — Outreach
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("PIPELINE 1 — HR OUTREACH")
print("=" * 70)

print("\nSubject:")
print(result["outreach"]["subject"])

print("\nEmail:")
print(result["outreach"]["email"])


# ---------------------------------------------------------
# 10. PIPELINE 2 — Job Analysis
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("PIPELINE 2 — JOB ANALYSIS")
print("=" * 70)

job_analysis = result["job_analysis"]

print("\nJob Role:")
print(job_analysis["job_role"])

print("\nMatch Percentage:")
print(f'{job_analysis["match_percentage"]}%')


print("\nStrong Matches:")

for skill in job_analysis["strong_matches"]:
    print(f"  ✓ {skill}")


print("\nPartial Matches:")

for skill in job_analysis["partial_matches"]:
    print(f"  ~ {skill}")


print("\nMissing Skills:")

for skill in job_analysis["missing_skills"]:
    print(f"  ✗ {skill}")


# ---------------------------------------------------------
# 11. LEARNING PLAN
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("LEARNING PLAN")
print("=" * 70)

for item in result["learning_plan"]:

    print("\nSkill:", item["skill"])
    print("Current Status:", item["current_status"])
    print("Priority:", item["priority"])
    print("Why Needed:", item["why_needed"])

    print("What to Learn:")

    for topic in item["what_to_learn"]:
        print(f"  - {topic}")

    print("Practical Task:")
    print(f"  {item['practical_task']}")


# ---------------------------------------------------------
# 12. RECOMMENDED LEARNING ORDER
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("RECOMMENDED LEARNING ORDER")
print("=" * 70)

for i, skill in enumerate(result["recommended_order"], 1):
    print(f"{i}. {skill}")