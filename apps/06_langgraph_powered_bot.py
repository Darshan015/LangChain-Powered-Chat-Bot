from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import Annotated
from langgraph.graph.message import add_messages

class chatState(BaseModel):
    messages:Annotated[list, add_messages]

llm = ChatGroq(model ="openai/gpt-oss-20b")
def chatBot(state:chatState) -> chatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

memory = InMemorySaver()

graph = StateGraph(chatState)
graph.add_node("chat_bot",chatBot)
graph.add_edge(START,"chat_bot")
graph.add_edge("chat_bot",END)
graph = graph.compile(checkpointer=memory)
config = {"configurable":{"thread_id":"my-bot-1"}}
while True:
    query = input("Enter your query")
    if query.lower() in ["quit","bye"]:
        print("Goodbye!!")
        break
    res = graph.invoke({"messages":[{"role":"user","content":query}]},config)
    print("AI:",res["messages"][-1].content)

