from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import asyncio
import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


class InputCGPA(BaseModel):
    marks: float = Field(description="Student marks")


class CGPATOOL(BaseTool):
    name: str = "cgpa_tool"
    description: str = "This tool converts marks to CGPA"
    args_schema: Type[BaseModel] = InputCGPA

    def _run(self, marks: float) -> float:
        return marks / 9.5

    async def _arun(self, marks: float) -> float:
        return marks / 9.5


cgpatool = CGPATOOL()


# Bind the tool to the LLM
llm_with_tool = llm.bind_tools([cgpatool])


# Ask the LLM
response = llm_with_tool.invoke(
    "Convert 43 marks to CGPA using the cgpa_tool"
)

print(response)