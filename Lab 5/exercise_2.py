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

class Review(BaseModel):
    research_questions: str = Field(description="research questions from the paper")
    method : str = Field(description="method os research")
    key_finding: str = Field(description="key findings from the research")


review_parser = PydanticOutputParser(pydantic_object=Review)

summary_parser = StrOutputParser()



input_prompt = PromptTemplate(
    template='''you are a helpuful assistant. extract research questions, method, key findings from this research paper given below.

{format_instructions}

Customer Review:
{research}''',
    input_variables=["research"],
    partial_variables={
        "format_instructions": review_parser.get_format_instructions()
    }
)
# Dummy review

dummy_research = """
A recent study investigated whether regular physical exercise improves concentration among college students. The researchers studied 100 students for four weeks. One group performed 30 minutes of exercise five days a week, while the other group maintained their normal routine. The results showed that students who exercised regularly had slightly better concentration scores. The researchers concluded that regular exercise may have a positive effect on students' concentration and academic performance.
"""


# Support ticket summary ChatPromptTemplate

Support_ticket_summary = ChatPromptTemplate.from_template(
    'generate plain-language summary for the non-expert audience \n {result}'
)


# Create the chain

chain = input_prompt | llm | review_parser | Support_ticket_summary | llm | summary_parser


# Invoke the chain

response = chain.invoke({"research": dummy_research})

print(response)