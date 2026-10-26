from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import Literal
from agent import editor_agent,get_llm,reasercher_agent, writer_agent
from state import BlogState

MAX_REVISION = 3
memory = InMemorySaver()

def reasearch_node(state:BlogState) -> BlogState:
    llm = get_llm()

    reasearch_data = reasercher_agent(
        llm = llm,
        topic= state.topic,
        audience= state.audience,
        feedback = state.reasearch_feedback

    )
    state.reasearch = reasearch_data
    state.reasearch_feedback = ""
    return state

def human_review_research_node(state:BlogState) ->BlogState:
    """ Pause and ask the user to review the reasearch or send the feedback"""
    decision = interrupt({
        "stage":"researcher_review",
        "reasearch" : state.reasearch,
        "instruction": ("Reply with 'approve' to continue to writing"
                        "or describe what changes to send it back to researcher")
    })
    if isinstance(decision,dict):
        action = decision.get("action","approve")
        feedback = decision.get("feedback","")

    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approved","approve","ok","yes",""] else "revise"
        feedback = "" if action=="approve" else text

    state.reasearch_feedback  = feedback
    return state

def draft_node(state:BlogState) -> BlogState:
    llm = get_llm()

    draft_data = writer_agent(
        llm = llm,
        topic= state.topic,
        audience= state.audience,
        feedback = state.draft_feedback

    )
    state.draft =  draft_data
    state.draft_feedback = ""
    return state

def human_review_draft_node(state:BlogState) ->BlogState:
    """ Pause and ask the user to draft the blog or send the feedback"""
    decision = interrupt({
        "stage":"draft_review",
        "draft" : state.draft,
        "instruction": ("Reply with 'approve' to continue to editor"
                        "or describe what changes to send it back to writer")
    })
    if isinstance(decision,dict):
        action = decision.get("action","approve")
        feedback = decision.get("feedback","")

    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approved","approve","ok","yes",""] else "revise"
        feedback = "" if action=="approve" else text

    state.draft_feedback  = feedback
    return state

def editor_node(state:BlogState) -> BlogState:
    llm = get_llm()

    final_editor_data = editor_agent(
        llm = llm,
        topic= state.topic,
        draft = state.draft
           

    )
    state.final_draft =  final_editor_data
    
    return state

#Conditional edge

def route_after_researcher_node(state:BlogState) -> Literal["researcher","writer"]:
    if state.reasearch_feedback:
        return "researcher"
    else:
        return "writer"

def route_after_writer_node(state:BlogState) -> Literal["editor","writer"]:
    if state.draft_feedback and state.revision_count < MAX_REVISION:
        return "writer"
    else:
        return "editor"
    
def add_graph():
    graph = StateGraph(BlogState)
    #define graph nodes
    graph.add_node("researcher",reasearch_node)
    graph.add_node("review_researcher", human_review_research_node)
    graph.add_node("writer", draft_node)
    graph.add_node("review_writer",human_review_draft_node)
    graph.add_node("editor",editor_node)

    #define edges
    graph.add_edge(START, "researcher")
    graph.add_edge( "researcher", "review_researcher")
    graph.add_conditional_edges( "review_researcher",route_after_researcher_node,{"researcher":"researcher", "writer":"writer"})
   
    graph.add_edge( "writer", "review_writer")
    graph.add_conditional_edges( "review_writer", route_after_writer_node, {"writer":"writer","editor":"editor" })
   
    graph.add_edge( "editor", END)

    graph = graph.compile(checkpointer=memory)

    return graph



