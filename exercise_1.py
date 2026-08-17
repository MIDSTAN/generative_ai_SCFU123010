# build a chatbot that recomends a collage club based on students intrests

# Required Functionalities
# 1. Accepts free text inputs describing the students intrests
# 2. recomends a specific club by name with a one line specific reason
# 2. runs in a loop until the user chooses to exit

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=api_key,
)

clubs = [
    "Coding Club",
    "Artificial Intelligence Club",
    "Robotics Club",
    "Data Science Club",
    "Cybersecurity Club",
    "Entrepreneurship Club",
    "Photography Club",
    "Music Club",
    "Dance Club",
    "Drama Club",
    "Debate Club",
    "Literary Club",
    "Art Club",
    "Sports Club",
    "Gaming Club",
    "Film & Media Club",
    "Environmental Club",
    "Science Club",
    "Social Service Club",
    "Cultural Club"
]

system_prompt = """
You are a college club recommendation assistant.

Your task is to recommend the most suitable college club based on the student's interests.

Instructions:
1. Carefully analyze the student's free-text description of their interests.
2. Recommend exactly ONE specific college club by name.
3. Give exactly ONE concise sentence explaining why that club matches the student's interests.
4. If the student's interests are broad or unclear, choose the club that best matches the strongest interest mentioned.
5. Do not recommend multiple clubs.
6. Keep the response concise and friendly.
7. Use this format:

Clubs that the university has : {clubs}
Recommended Club: [<Club Name>]
Reason: <One-line specific reason>

Example:
Student: I enjoy programming, building applications, and learning about artificial intelligence.

Recommended Club: Coding & AI Club
Reason: This club matches your interest in programming and building AI-based applications.

This is the user's prompt : 
"""




while True:
    prompt = input("Specify your field of intrests : ")

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
        ],
        model="openai/gpt-oss-20b",
        )
    response = chat_completion.choices[0].message.content


    print(response)

    end = input("Continue? (True/False): ").lower()

    if end == "false":
        break



