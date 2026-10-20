from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()
search = GoogleSerperAPIWrapper()
st.subheader("Get your answer faster than speed of light with Groq")

if "memory" not in st.session_state:
   st.session_state.memory = MemorySaver()
   st.session_state.history = []

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


model = ChatGroq(model = "openai/gpt-oss-20b", streaming=True)

agent = create_agent(
    model=model,
    tools = [search.run],
    checkpointer= st.session_state.memory,
    system_prompt= "You are a high speed chat bot should provide the result efficiently with high speed"

)
query = st.chat_input("Shoot anything!")
if query:

    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    response = agent.stream(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"session_001"}},
        stream_mode="messages")

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        message = ""
        for chunks in response:
            message = message+chunks[0].content
            space.write(message)
    st.session_state.history.append({"role":"ai","content":message})
