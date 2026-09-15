import os
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated

def cgpa(marks: float) -> float:
    return round(marks / 9.5, 2)

def sgpa(marks: float) -> float:
    return round(marks / 9.5, 2)

def percentage(marks: float) -> float:
    return round(marks, 2)

def percentile(marks: float) -> float:
    return round(marks, 2)

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


class InputMarks(BaseModel):
    marks : float  = Field(description="this function converts marks to cgpa")
    
func = [cgpa, sgpa, percentage, percentile]
name = ["cgpa", "sgpa", "percentage", "percentile"]
descrip = ["this function converts marks to cgpa","this function converts marks to sgpa","this function converts marks to percentage","this function converts marks to percentile"]
    
tool_cgpa = StructuredTool(
    func=cgpa,
    name="cgpa",
    description="Convert the given percentage marks into CGPA. Use this tool whenever the user asks for CGPA.",
    args_schema=InputMarks
)

tool_percentage = StructuredTool(
    func=percentage,
    name="percentage",
    description="Return the given marks as percentage. Use this tool whenever the user asks for percentage.",
    args_schema=InputMarks
)
llm_with_tool=llm.bind_tools([tool_cgpa,tool_percentage])
response = llm_with_tool.invoke(
    "Convert 95 marks into percentage and CGPA. You must call both the percentage and cgpa tools."
)

print(response.tool_calls)
# print()

# for i in range(4):
#     structured_tool = StructuredTool(
#         func = func[i],
#         name = name[i],
#         description = descrip[i],
#         args_schema = InputMarks
#     )

#     llm_with_tool = llm.bind_tools([structured_tool])
#     print(structured_tool.invoke({"marks": 100}))