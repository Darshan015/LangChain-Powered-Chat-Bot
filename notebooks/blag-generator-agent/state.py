from pydantic import BaseModel, Field

class BlogState(BaseModel):
    #Usre input
    topic:str = ""
    audience:str = "general_audience"

    #Reasercher output
    reasearch:str = ""
    reasearch_feedback:str = ""

    #Writer Output
    draft:str = ""
    draft_feedback:str = ""

    #Editor output
    final_draft:str = ""

    #meta data
    revision_count:int = 0

    #