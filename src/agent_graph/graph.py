"""LangGraph definition with memory support."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_graph.state import InternalState, InputState, OutputState
from agent_graph.nodes import clarifier_node, researcher_node, summarizer_node, should_continue

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

import logging

# Configure the logger (optional but recommended)
logging.basicConfig(
    level=logging.INFO,                     # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# Create a logger instance
logger = logging.getLogger(__name__)

set_llm_cache(SQLiteCache(database_path="explorer_cache.db"))


def create_graph(
    with_checkpointer: bool = True
):
    """
    Creates and compiles the agent graph.
    
    Args:
        with_checkpointer: If True, enables memory with SQLite checkpointer
        llm_model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
        llm_temperature: Temperature for LLM (0 = deterministic, 1 = creative)
        max_papers: Maximum number of papers to retrieve from ArXiv
        max_iterations: Maximum retry attempts if not enough papers
    
    Note: These parameters are stored in the initial state's 'config' field
          and read by nodes during execution.
    """
    
    workflow = StateGraph(
        InternalState,        # Internal working state
        input=InputState,     # Only accept 'query' from users
        output=OutputState    # Only return 'summary', 'papers', 'messages'
    )
    
    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("summarizer", summarizer_node)
    
    workflow.set_entry_point("clarifier")
    workflow.add_edge("clarifier", "researcher")
    workflow.add_edge("researcher", "summarizer")
    
    workflow.add_conditional_edges(
        "summarizer",
        should_continue,
        {
            "researcher": "researcher",
            "end": END,
        }
    )
    
    if with_checkpointer:
        import sqlite3
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory = SqliteSaver(conn)
        
        print("✅ Graph compiled with memory enabled")
        return workflow.compile(checkpointer=memory)
    else:
        print("✅ Graph compiled without memory")
        return workflow.compile()
    
# Default graph instance for LangGraph Studio
graph = create_graph(with_checkpointer=False)