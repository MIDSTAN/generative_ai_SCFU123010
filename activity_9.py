import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=api_key,
)

history = []

while True:
    prompt = input("Ask Anything: ")

    # Add user's message to history
    history.append({
        "role": "user",
        "content": prompt,
    })

    chat_completion = client.chat.completions.create(
        messages=history,
        model="openai/gpt-oss-20b",
    )

    response = chat_completion.choices[0].message.content

    # Add AI response to history
    history.append({
        "role": "assistant",
        "content": response,
    })

    print(response)

    end = input("Continue? (True/False): ").lower()

    if end == "false":
        break


print(history)