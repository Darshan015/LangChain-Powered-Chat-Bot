from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

search = GoogleSerperAPIWrapper()
memory = MemorySaver()
model = ChatGroq(model = "openai/gpt-oss-20b", streaming=True)
agent = create_agent(
    model = model,
     tools = [search.run],
     checkpointer=memory,
    system_prompt= "You are a serach engine which search query from google using tool calling" )

while True:
    query = input("Query:" )
    if query.lower() in ["exit","bye"]:
        print("Goodbye!!")
        break


    response = agent.invoke({"messages":[{"role":"user", "content":query}]},
                            {"configurable": {"thread_id": "session_abc_12"}})
    print(response["messages"][-1].content)