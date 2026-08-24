
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

customer_querry = """
the tripod is buyed was damaged
"""
    
prompt = """
role : you are a Customer Support reply generator expert
context : you are an AI Customer Chat support agent
task : look at customer's query and answer their querries 
input data : {customer_querry}
output format: output should be in a kind and curious manner 
"""


# Insert resume and required fields into the existing prompt
final_prompt = prompt.format(customer_querry=customer_querry)

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