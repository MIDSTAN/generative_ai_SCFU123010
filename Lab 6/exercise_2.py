import os

from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


load_dotenv()

class InputDetails(BaseModel):
    principle: float = Field(
        description="Principal amount of the loan"
    )

    interest: float = Field(
        description="Annual interest rate in percentage"
    )

    tenure: int = Field(
        description="Tenure of loan in months"
    )

    @field_validator("principle", "interest", "tenure")
    @classmethod
    def validate_input(cls, v):
        if v <= 0:
            raise ValueError("Invalid entry. Value must be greater than 0.")
        return v

def calculate_emi(
    principle: float,
    interest: float,
    tenure: int
) -> float:

    monthly_interest = interest / (12 * 100)

    emi = (
        principle
        * monthly_interest
        * (1 + monthly_interest) ** tenure
    ) / (
        (1 + monthly_interest) ** tenure - 1
    )

    return emi

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

tool_EMI = StructuredTool(
    func=calculate_emi,
    name="calculate_emi",
    description=(
        "Calculate EMI from the given loan principal, "
        "annual interest rate, and tenure in months."
    ),
    args_schema=InputDetails
)

llm_with_tool = llm.bind_tools([tool_EMI])

response = llm_with_tool.invoke(
    "What's my EMI for a 5 lakh loan at 9% for 24 months?"
)


print(response.tool_calls)