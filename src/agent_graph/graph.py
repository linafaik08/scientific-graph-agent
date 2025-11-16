"""LangGraph definition with memory support."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_graph.state import InternalState, InputState, OutputState
from agent_graph.nodes import (
    clarifier_node,
    researcher_node,
    summarizer_node,
    should_continue,
    researcher_node_streaming,
    summarizer_node_streaming,
    approver_node,
    route_after_approval
)

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
    with_checkpointer: bool = True,
    interrupt_before: list[str] = [],
    interrupt_after: list[str] = [],
):
    """
    Creates and compiles the agent graph with optional interrupts.
    
    Args:
        with_checkpointer: Enable state persistence for interrupts
        interrupt_before: Node names to pause BEFORE execution
        interrupt_after: Node names to pause AFTER execution
    """
    
    workflow = StateGraph(
        InternalState,
        input=InputState,
        output=OutputState,
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
        {"researcher": "researcher", "end": END}
    )
    
    if with_checkpointer:
        import sqlite3
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory = SqliteSaver(conn)
        return workflow.compile(
            checkpointer=memory,
            interrupt_before=interrupt_before,  # 👈 Pause before these nodes
            interrupt_after=interrupt_after,     # 👈 Pause after these nodes
        )
    else:
        return workflow.compile(
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after
        )

async def create_streaming_graph(
    with_checkpointer: bool = True
):
    """
    Creates and compiles the agent graph with streaming support.

    This version uses async nodes that support:
    - Token-level streaming for LLM outputs (summarizer)
    - Custom progress updates (ArXiv search)
    - Node-level state updates

    Args:
        with_checkpointer: If True, enables memory with AsyncSqliteSaver

    Usage:
        Use with astream() and stream_mode parameter:
        - stream_mode="updates": Get state updates after each node
        - stream_mode="messages": Get LLM token-level updates
        - stream_mode="custom": Get custom progress updates from tools
        - stream_mode=["updates", "messages", "custom"]: Combine all modes

    Example:
        ```python
        graph = await create_streaming_graph(with_checkpointer=True)

        async for event in graph.astream(
            initial_state,
            config=config,
            stream_mode=["updates", "custom"]
        ):
            print(event)
        ```

    Note:
        Requires aiosqlite package for async checkpointing.
        Install with: pip install aiosqlite
    """

    workflow = StateGraph(
        InternalState,        # Internal working state
        input=InputState,     # Only accept 'query' from users
        output=OutputState    # Only return 'summary', 'papers', 'messages'
    )

    # Use async streaming versions of the nodes
    workflow.add_node("clarifier", clarifier_node)  # Clarifier doesn't need streaming
    workflow.add_node("researcher", researcher_node_streaming)
    workflow.add_node("summarizer", summarizer_node_streaming)

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
        import aiosqlite

        # Create async connection for AsyncSqliteSaver
        conn = await aiosqlite.connect(":memory:")
        memory = AsyncSqliteSaver(conn)

        logger.info("✅ Streaming graph compiled with async memory enabled")
        return workflow.compile(checkpointer=memory)
    else:
        logger.info("✅ Streaming graph compiled without memory")
        return workflow.compile()

def create_graph_with_approval(with_checkpointer: bool = True):
    """
    Creates a graph with a built-in approval node that demonstrates
    node-level interrupts.
    """
    
    workflow = StateGraph(
        InternalState,
        input=InputState,
        output=OutputState
    )
    
    # Add all our nodes, including the new approval node
    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("approver", approver_node)  # 👈 New approver gate
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("summarizer", summarizer_node)
    
    # Flow: clarifier → approver → researcher (or end)
    workflow.set_entry_point("clarifier")
    workflow.add_edge("clarifier", "approver")
    
    workflow.add_conditional_edges(
        "approver",
        route_after_approval,
        {"researcher": "researcher", "end": END}
    )
    
    workflow.add_edge("researcher", "summarizer")
    workflow.add_conditional_edges(
        "summarizer",
        should_continue,
        {"researcher": "researcher", "end": END}
    )
    
    if with_checkpointer:
        import sqlite3
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory = SqliteSaver(conn)
        return workflow.compile(checkpointer=memory)
    
    return workflow.compile()

# Default graph instance for LangGraph Studio
# graph = create_graph(with_checkpointer=False)
graph = create_graph_with_approval(with_checkpointer=False)