# Review Classifier

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

resume_text = """
Pruthviraj Santosh Sapate
Solapur, Maharashtra, India | +91 76666 73289 | pruthvirajsapate123@gmail.com
GitHub | LinkedIn

Career Objective
As a third-year Computer Science student, I’m excited to build on my solid full stack development
skills while diving deeper into image processing. I’ve enjoyed creating web apps that feel intuitive
and powerful, and now I’m keen to tackle projects where AI meets real-world visuals—like detecting
patterns in images that could change how we analyze data. I’m looking for a spot on a team that’s
pushing boundaries, where I can learn, contribute my code, and grow into someone who bridges frontend
flair with backend smarts and computer vision magic.

Technical Skills
Languages: C, C++, Python, JavaScript
Frameworks/Libraries: Flask, React.js, Tailwind CSS, YOLO, Django
Databases: PostgreSQL, MySQL
Tools: Git, Postman

Projects
• Employee Work Time Detection – Python, YOLO
In a intense 24-hour hackathon in Vijayapura, I put together a system that uses YOLO to watch
video feeds and spot if employees are focused or taking a break.

• RAG-Based Physicist Chatbot – React.js, Flask/Django, Tailwind CSS
During my internship, I crafted this chatbot that pulls from a physicist’s decades of research
to field questions in plain English.

Internship Experience
E-arth Solutions Pvt Ltd July 2024 – Sept 2024
Web & AI Developer Intern Remote

Education
B.Tech in Computer Science Engineering CGPA: 7.1 (No Backlogs)
MIT Vishwaprayag University, Kegaon, Solapur, Maharashtra 2023 – 2027
HSC (Science) 56%
Pune Board
"""

required_fields = [
    "Name",
    "Email",
    "Phone",
    "Location",
    "Career Objective",
    "Technical Skills",
    "Programming Languages",
    "Frameworks and Libraries",
    "Databases",
    "Tools",
    "Projects",
    "Internship Experience",
    "Education",
    "CGPA",
    "HSC Percentage"
]

prompt = """
role : you are a resume field extractor expert
context : extract required fields fro the resume text
task : look at he resume text and the given fields, and find those given fields in the given resume text
input data : {resume_text}{required_fields}
output format: Strict json fromat only and the json should contain some info about the resume person and given fields info about the user
examples :
"""

# Examples required by the prompt
examples = """
Example 1:
Resume:
John Doe
Mumbai, India
john@gmail.com
B.Tech Computer Science
CGPA: 8.2

Required Fields:
["Name", "Email", "Location", "Education", "CGPA"]

Expected JSON:
{
    "Name": "John Doe",
    "Email": "john@gmail.com",
    "Location": "Mumbai, India",
    "Education": "B.Tech Computer Science",
    "CGPA": "8.2"
}

Example 2:
Resume:
Jane Smith
Python, Java, Django
ABC Technologies
Software Developer Intern
B.Tech IT
CGPA: 8.7

Required Fields:
["Name", "Programming Languages", "Internship Experience", "Education", "CGPA"]

Expected JSON:
{
    "Name": "Jane Smith",
    "Programming Languages": ["Python", "Java"],
    "Internship Experience": "Software Developer Intern at ABC Technologies",
    "Education": "B.Tech IT",
    "CGPA": "8.7"
}
"""

# Insert resume and required fields into the existing prompt
final_prompt = prompt.format(
    resume_text=resume_text,
    required_fields=json.dumps(required_fields)
)

# Add examples without modifying the original prompt
final_prompt += "\n" + examples

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": final_prompt
        }
    ],
    temperature=0
)

result = response.choices[0].message.content

print(result)