import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

#define LLM agent

def get_llm(model_name:str =  "openai/gpt-oss-20b", temperature:float = 0.5):
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=model_name,
                   temperature=temperature,
                   api_key=api_key)
    return llm

#Reasearch agent

reasercher_prompt = ChatPromptTemplate.from_messages([
    {"role":"system", "content": """
    "You are a reasearch agent, given a blog topic and audience, "
    "produce a clear and structured reaserach outline, Include "\n"
      "1. 5-7 key points that the blog should cover"
      "2. Important facts, stats for the keypoints"
      "3. Suggested angle or hook"
    "Be concise, use bullet points and DO NOT write the full blog yet"
    
    """},
    {"role":"user", "content":"Topic :{topic} Audience: {audience} {revision_hints} Write the reaserach outline now"}
])

def reasercher_agent(llm:ChatGroq, topic:str, audience:str, feedback:str = "") ->str:
    revision_hints = f"User provided this feedback on the previous research, please address it: {feedback}"
    if not revision_hints:
        revision_hints = f"This is your firs attempt"

    chain = reasercher_prompt | llm
    result = chain.invoke({
        "topic":topic,
        "audience": audience,
        "revision_hints" :revision_hints
    })

    return result.content

#Writer agent

writer_prompt = ChatPromptTemplate.from_messages([
    {"role":"system", "content": """
    "You are a Blog Writer agent, using the reaserch note provided, Write "
    "a complete engaging blog post, Include "\n"
      "1. Length:500-800 words"
      "2. Structure: Catchy title, intro hook and 3-5 section with H2 heading, conclusion \n"
      "3. Tone: clear, friendly, suited to the target audience"
      "4. Use markdown formatting\n"
      "5. Do NOT add wordcount line at the end"
    
    
    """},
    {"role":"user", "content":"Topic :{topic} Audience: {audience} {revision_hints} Write the Complete blog now"}
])

def writer_agent(llm:ChatGroq, topic:str, audience:str, reasearch:str ="", feedback:str = "") ->str:
    revision_hints = f"User provided this feedback on the previous draft, and asked for these changes: {feedback} Please apply these changes during writing the blog"
    if not revision_hints:
        revision_hints = f"This is your firs attempt"

    chain = writer_prompt | llm
    result = chain.invoke({
        "topic":topic,
        "audience": audience,
        "research": reasearch,
        "revision_hints" :revision_hints
    })

    return result.content

#Editor agent
editor_prompt = ChatPromptTemplate.from_messages([
    {"role":"system", "content": """
    "You are an Editor agent - the final quality gate before publishing.\n"
    "Take the draft and produce the FINAL polished vesrion. Specifically: \n"
    "- Fix grammer, spelling, and awkward phrasing \n"
    "-Tighten wordy sentences\n"
    "-Improve flow and transition between section\n"
    "-Make teh title and intro more compelling if needed\n"
    "-Keep the same structure and markdown formatting\n"
    "Output only the final polished blog post - no commentary."
    
    
    """},
    {"role":"user", "content":"Topic :{topic} Draft:{draft} Write the Complete blog now"}
])

def editor_agent(llm:ChatGroq, topic:str, draft:str = "") ->str:

    chain = editor_prompt | llm
    result = chain.invoke({
        "topic":topic,      
        "draft" :draft
    })

    return result.content


