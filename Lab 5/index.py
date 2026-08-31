from langchain_groq import ChatGroq
import os
import json
from dotenv import load_dotenv
from groq import Groq
# from langsmith import Client
# from langchain_core.prompts import ChatPromptTemplate

# client = Client()

# question = """what is the meaning of india"""

# prompt = ChatPromptTemplate([
#     ("system", "You are a helpful chatbot."),
#     ("user", "{question}"),
# ])

# print(prompt.invoke({"question":question}))

from langchain_core.prompts import PromptTemplate

# Instantiation using from_template (recommended)
prompt = PromptTemplate.from_template("Say {foo}")
prompt.format(foo="bar")

person = """
Erode Venkatappa Ramasamy, commonly known as Periyar
"""


# Instantiation using initializer
prompt = PromptTemplate(template="write a poem on {person}, return only the poem")

load_dotenv()
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

print(llm.invoke(prompt.invoke({"person":person})))