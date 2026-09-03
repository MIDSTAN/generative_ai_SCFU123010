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
    core_complaint: str = Field(description="Actual Complaint of the customer")
    product_feature_mentioned: str = Field(description="Feature of the purchased product")
    customer_sentiment: str = Field(description="Sentiment of the customer")


review_parser = PydanticOutputParser(pydantic_object=Review)

summary_parser = StrOutputParser()



input_prompt = PromptTemplate(
    template='''you are a helpuful assistant. extract core complaint, product/feature mentioned, and customer sentiment from this customer rivew given below.

{format_instructions}

Customer Review:
{person}''',
    input_variables=["person"],
    partial_variables={
        "format_instructions": review_parser.get_format_instructions()
    }
)
# Dummy review

dummy_review = """
I purchased a wireless Bluetooth headphone last week, but the battery
only lasts for about two hours instead of the advertised ten hours.
The sound quality is good, but the poor battery life is very disappointing.
"""


# Support ticket summary ChatPromptTemplate

Support_ticket_summary = ChatPromptTemplate.from_template(
    'you are a helpuful assistant. generate ticket summary for this structured data given below \n {result}'
)


# Create the chain

chain = input_prompt | llm | review_parser | Support_ticket_summary | llm | summary_parser


# Invoke the chain

response = chain.invoke({"person": dummy_review})

print(response)