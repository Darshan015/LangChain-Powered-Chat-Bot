from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
import streamlit as st

#PDF document loading

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):
    doc_load = PyPDFDirectoryLoader(path)
    document_load = doc_load.load()

    #Document splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    document = splitter.split_documents(documents=document_load)

    #Embedding and vectore store
    embedding = OpenAIEmbeddings(model = "text-embedding-3-large")
    vector_store = InMemoryVectorStore.from_documents(documents = document,
                                                    embedding=embedding)
    
    #query = "What is the name of the patient?"
    llm = ChatGroq(model = "openai/gpt-oss-20b")

    @tool
    def retrive_context(query:str):
        """
        This is the tool to extract the external document and provide accurate results
        """
        data = vector_store.similarity_search(query=query, k = 3)
        context = ""

        for doc in data:
            context+=doc.page_content +"\n"
        return context
    
    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retrive_context` tool for questions requiring external knowledge."""
    memory = InMemorySaver()
    agent = create_agent(model = llm,
                            tools = [retrive_context],
                            system_prompt=system_prompt,
                            checkpointer=memory
                            )
    st.session_state.document_uploaded = True
    st.session_state.agent =agent

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Please upload your PDF file",type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..!"):
            path = "./doc_files/"
        
            for file in uploaded:
                with open(path+file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)
      


    query = st.chat_input("Ask anything..")
    if query:
        st.session_state.messages.append({"role":"user","content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke({"messages":[{"role":"user","content":query}]},
                                                 {"configurable":{"thread_id":"1"}})
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})




    



