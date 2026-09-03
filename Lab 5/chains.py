import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Create the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


# Create the output parser
parser = StrOutputParser()


# Create the prompt
input_prompt = PromptTemplate(template='you are a helpuful assistant. generate poem about this {person}', input=["person"])

# input_1  = input_prompt.invoke("virat kolhi")

# result = llm.invoke(input)


summary_prompt = PromptTemplate(template='you are a helpuful assistant. generate summary for this poem {result}', input=["result"])

# summary  = summary_prompt.invoke(result)

# Create the chain
chain = input_prompt | llm | parser | summary_prompt | llm | parser
# Invoke the chain
response = chain.invoke("periyar")

print(response)
