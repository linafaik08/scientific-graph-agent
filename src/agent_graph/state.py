"""Shared state between graph nodes."""
from typing import TypedDict, List, Annotated
from typing import NotRequired
from operator import add
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

# === REDUCERS ===

def keep_top_papers(existing: List[dict], new: List[dict], n_top: int) -> List[dict]:
    """
    Reducer function to maintain top N papers by relevance score.

    Combines existing and new papers, removes duplicates (keeping highest score),
    and returns the top N papers sorted by relevance.

    Args:
        existing: Previously stored papers
        new: Newly added papers
        n_top: Number of top papers to keep

    Returns:
        Top N papers sorted by relevance score
    """
    combined = existing + new
    
    # Remove duplicates, keeping highest score
    seen = {}
    for paper in combined:
        paper_id = paper['id']
        if paper_id not in seen or paper['relevance_score'] > seen[paper_id]['relevance_score']:
            seen[paper_id] = paper
    
    # Sort by relevance, keep top 10
    sorted_papers = sorted(seen.values(), 
                          key=lambda p: p.get('relevance_score', 0), 
                          reverse=True)
    return sorted_papers[:n_top]

def summarize_messages(
        existing: List[BaseMessage],
        new: List[BaseMessage],
        min_meesage: int = 3,
        llm_model_name = "gpt-4o-mini",
        llm_temperature: int = 1,
        ) -> List[BaseMessage]:
    """
    Reducer function to summarize conversation history when it gets too long.

    Keeps the most recent messages and summarizes older ones to save context.

    Args:
        existing: Previously stored messages
        new: Newly added messages
        min_meesage: Minimum number of recent messages to keep (default: 3)
        llm_model_name: Model to use for summarization (default: gpt-4o-mini)
        llm_temperature: Temperature for summarization (default: 1)

    Returns:
        Summarized message history with recent messages preserved
    """
    combined = existing + new

    # Keep last min_meesage messages as-is
    if len(combined) <= min_meesage:
        return combined

    # Summarize everything before the last 10
    old_messages = combined[:-min_meesage]
    recent_messages = combined[-min_meesage:]

    llm = ChatOpenAI(model=llm_model_name, temperature=llm_temperature)

    # Build proper message list for LLM
    messages_to_summarize = [
        SystemMessage(content="Summarize this conversation history concisely in 2-3 sentences.")
    ] + old_messages

    # Get summary response
    summary_response = llm.invoke(messages_to_summarize)

    # Extract the text content from the AIMessage response
    summary_text = summary_response.content

    return [SystemMessage(content=summary_text)] + recent_messages

# === PUBLIC INTERFACE ===

class InputState(TypedDict):
    """Input state for the graph - user-provided query."""
    query: str  # User question (REQUIRED)
    llm_model: NotRequired[str]  # Language model to use
    llm_temperature: NotRequired[float]  # Temperature for LLM calls
    max_papers: NotRequired[int]  # Maximum papers to retrieve
    max_iterations: NotRequired[int]  # Maximum refinement iterations

class OutputState(TypedDict):
    """Output state for the graph - results returned to user."""
    summary: str  # Final summary with bullet points
    papers: Annotated[List[dict], lambda e, n: keep_top_papers(e, n, n_top=10)]  # Papers found from ArXiv
    messages: Annotated[List[BaseMessage], lambda e, n: summarize_messages(
        e, n,
        min_meesage=3,
        llm_model_name="gpt-4o-mini",
        llm_temperature=1)]  # Conversation history

# === INTERNAL WORKING STATE ===

class InternalState(InputState, OutputState):
    """
    Full state used internally by nodes.
    """
    refined_query: NotRequired[str]  # Refined query produced by clarifier
    iteration: NotRequired[int]  # Loop counter for retry logic
    
class PrivateState(TypedDict):
    """Private state for internal node processing."""
    refined_query: str  # Refined query after clarification
    iteration: int  # Loop counter to avoid infinite loops
