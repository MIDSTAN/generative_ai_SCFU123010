
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

language = """
russian
japanese
hindi 
kannad
telgu 
gujrati
french
chienese
"""

user_input="""
काय राव, एवढं काय घाबरता?
"""

    
prompt = """
role : you are a Multi language translator expert
context : you have to translate customer input into given {language}
task : take users input and identify its language and then translate it into user dersired {language} 
input data : {user_input}
output format: output should be strictly in same text as the language that user wants to translate in, if multiple languages are mentioned translate in multiple languages
"""


# Insert resume and required fields into the existing prompt
final_prompt = prompt.format(language=language, user_input = user_input)

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