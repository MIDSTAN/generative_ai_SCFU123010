from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
import json
from dotenv import load_dotenv
from groq import Groq

parser = StrOutputParser()
load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

# Get string output from a model
message = llm.invoke("Tell me a joke")
result = parser.invoke(message)
print(result)  # plain string
