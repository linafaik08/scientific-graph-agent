"""Shared state between graph nodes."""
from typing import TypedDict, List, Annotated
from operator import add
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """State of the agent graph.
    
    Uses LangChain message types for proper integration with LangSmith.
    Only 'query' is required, all other fields have defaults.
    """
    # Required fields
    query: str                              # User question (REQUIRED)
    
    # Optional fields with defaults
    refined_query: str                      # Refined query after clarification
    papers: List[dict]                      # Papers found from ArXiv
    summary: str                            # Final summary with bullet points
    iteration: int                          # Loop counter to avoid infinite loops
    config: dict                            # Graph configuration (llm, max_papers, etc.)
    messages: Annotated[List[BaseMessage], add]  # Conversation history