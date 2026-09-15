from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
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

addition.invoke({"a": 4, "b": 6})

llm_with_tools = llm.bind_tools([addition])

print(llm_with_tools.invoke("add 4 and 6"))

message_1 = []

def addition_agent(query):
    message_1.append(HumanMessage(content=query))

    print(message_1)

    ai_response = llm_with_tools.invoke(message_1)

    print("step 1 : ", ai_response)

    if ai_response.tool_calls:
        message_1.append(ai_response)

        for tool_call in ai_response.tool_calls:
            tool_response = addition.invoke(tool_call["args"])

            print("step 2 : ", tool_response, tool_call["id"])

            message_1.append(
                ToolMessage(
                    content=str(tool_response),
                    tool_call_id=tool_call["id"]
                )
            )

        final_response = llm_with_tools.invoke(message_1)

        print("step 3 : ", final_response)

        message_1.append(final_response)

        return final_response.content

    return ai_response.content


result = addition_agent("Add 15 and 25")
print("Final answer : ", result)