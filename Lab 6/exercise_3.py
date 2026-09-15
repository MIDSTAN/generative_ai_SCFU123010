import os

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


load_dotenv()

@tool
def convert_temp(
    celsius: float,
) -> float:
    """Converts Temprature in Celsius to Fahrenheit"""
    return (celsius * 1.8) + 32

@tool
def convert_dist(
    km: float,
) -> float:
    """Converts distance in km to miles"""
    return km * 0.62

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


llm_with_tool = llm.bind_tools([convert_temp, convert_dist])

response = llm_with_tool.invoke(
    "What's is 36 km in miles?"
)


print(response.tool_calls)