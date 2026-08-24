# Review Classifier

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

with open("reviews_with_sentiments.json", "r") as file:
    data = json.load(file)

reviews = "this is a very bad product"

prompt = """
role : you are a costumer review sentiment analysis expert
context : you will recieve reviews of customers and you need to analyze its sentiments
task : do sentiment analysis of the given sentences, use these four categories Positive, Negative, Mixed and Neutral
input data : {reviews}
output format: Review ad its sentiment side by side
examples :
"""

# Insert the review into the prompt
final_prompt = prompt.format(reviews=reviews)

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