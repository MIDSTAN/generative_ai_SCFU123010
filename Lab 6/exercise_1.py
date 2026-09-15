from langchain_core.tools import tool
import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


@tool
def shipping_status(order_id: str) -> str:
    """Get the shipping status of an order given its order ID."""
    return f"Order {order_id} is currently out for delivery."

print(shipping_status.invoke({"order_id": "ORD-4521"}))

llm_with_tools = llm.bind_tools([shipping_status])


response = llm_with_tools.invoke(
    "Where is my order ORD-4521?"
)

print(response)