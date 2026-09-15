from langchain_core.tools import tool
import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

@tool
def addition(a: int, b: int) -> int:
    """This tool gives the addition of a and b."""
    return a + b

addition.invoke({"a":4, "b":6})

llm_with_tools = llm.bind_tools([addition])

print(llm_with_tools.invoke("multiply these numbers"))