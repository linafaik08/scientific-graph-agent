"""LangGraph definition with memory support."""
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Send

from agent_graph.state import InternalState, InputState, OutputState
from agent_graph.nodes import (
    clarifier_node,
    arxiv_researcher_node,
    wikipedia_researcher_node,
    arxiv_researcher_node_streaming,
    wikipedia_researcher_node_streaming,
    summarizer_node,
    should_continue,
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
    tools: list[str] = ["arxiv"],
    mode: str = "sequential"
):
    """
    Creates and compiles the agent graph with configurable tools and execution mode.

    Args:
        with_checkpointer: Enable state persistence for interrupts
        interrupt_before: Node names to pause BEFORE execution
        interrupt_after: Node names to pause AFTER execution
        tools: List of research tools to use (e.g., ["arxiv"], ["wikipedia"], ["arxiv", "wikipedia"])
        mode: Execution mode - "sequential" (one after another) or "parallel" (all at once)
    """

    # Map tool names to their node functions
    tool_node_map = {
        "arxiv": ("arxiv_researcher", arxiv_researcher_node),
        "wikipedia": ("wikipedia_researcher", wikipedia_researcher_node)
    }

    workflow = StateGraph(
        InternalState,
        input=InputState,
        output=OutputState,
    )

    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("summarizer", summarizer_node)

    # Add researcher nodes for each requested tool
    researcher_nodes = []
    for tool in tools:
        if tool in tool_node_map:
            node_name, node_func = tool_node_map[tool]
            workflow.add_node(node_name, node_func)
            researcher_nodes.append(node_name)
        else:
            logger.warning(f"Unknown tool '{tool}' requested, skipping")

    if not researcher_nodes:
        raise ValueError("No valid research tools specified. Use 'arxiv' or 'wikipedia'.")

    workflow.set_entry_point("clarifier")

    if mode == "parallel":
        # In parallel mode, all researchers run simultaneously from clarifier
        for node_name in researcher_nodes:
            workflow.add_edge("clarifier", node_name)
            workflow.add_edge(node_name, "summarizer")

        logger.info(f"✅ Graph configured with parallel execution: {', '.join(researcher_nodes)}")
    else:  # sequential mode
        # In sequential mode, researchers run one after another
        workflow.add_edge("clarifier", researcher_nodes[0])

        for i in range(len(researcher_nodes) - 1):
            workflow.add_edge(researcher_nodes[i], researcher_nodes[i + 1])

        workflow.add_edge(researcher_nodes[-1], "summarizer")

        logger.info(f"✅ Graph configured with sequential execution: {' → '.join(researcher_nodes)}")

    workflow.add_conditional_edges(
        "summarizer",
        should_continue,
        {researcher_nodes[0]: researcher_nodes[0], "end": END}
    )

    if with_checkpointer:
        import sqlite3
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory = SqliteSaver(conn)
        return workflow.compile(
            checkpointer=memory,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )
    else:
        return workflow.compile(
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after
        )

async def create_streaming_graph(
    with_checkpointer: bool = True,
    tools: list[str] = ["arxiv"],
    mode: str = "sequential"
):
    """
    Creates and compiles the agent graph with streaming support and configurable tools.

    This version uses async nodes that support:
    - Token-level streaming for LLM outputs (summarizer)
    - Custom progress updates (ArXiv/Wikipedia search)
    - Node-level state updates

    Args:
        with_checkpointer: If True, enables memory with AsyncSqliteSaver
        tools: List of research tools to use (e.g., ["arxiv"], ["wikipedia"], ["arxiv", "wikipedia"])
        mode: Execution mode - "sequential" (one after another) or "parallel" (all at once)

    Usage:
        Use with astream() and stream_mode parameter:
        - stream_mode="updates": Get state updates after each node
        - stream_mode="messages": Get LLM token-level updates
        - stream_mode="custom": Get custom progress updates from tools
        - stream_mode=["updates", "messages", "custom"]: Combine all modes

    Example:
        ```python
        graph = await create_streaming_graph(
            with_checkpointer=True,
            tools=["arxiv", "wikipedia"],
            mode="parallel"
        )

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

    # Map tool names to their async node functions
    tool_node_map = {
        "arxiv": ("arxiv_researcher", arxiv_researcher_node_streaming),
        "wikipedia": ("wikipedia_researcher", wikipedia_researcher_node_streaming)
    }

    workflow = StateGraph(
        InternalState,        # Internal working state
        input=InputState,     # Only accept 'query' from users
        output=OutputState    # Only return 'summary', 'papers', 'messages'
    )

    # Use async streaming versions of the nodes
    workflow.add_node("clarifier", clarifier_node)  # Clarifier doesn't need streaming
    workflow.add_node("summarizer", summarizer_node_streaming)

    # Add researcher nodes for each requested tool
    researcher_nodes = []
    for tool in tools:
        if tool in tool_node_map:
            node_name, node_func = tool_node_map[tool]
            workflow.add_node(node_name, node_func)
            researcher_nodes.append(node_name)
        else:
            logger.warning(f"Unknown tool '{tool}' requested, skipping")

    if not researcher_nodes:
        raise ValueError("No valid research tools specified. Use 'arxiv' or 'wikipedia'.")

    workflow.set_entry_point("clarifier")

    if mode == "parallel":
        # In parallel mode, all researchers run simultaneously from clarifier
        for node_name in researcher_nodes:
            workflow.add_edge("clarifier", node_name)
            workflow.add_edge(node_name, "summarizer")

        logger.info(f"✅ Streaming graph configured with parallel execution: {', '.join(researcher_nodes)}")
    else:  # sequential mode
        # In sequential mode, researchers run one after another
        workflow.add_edge("clarifier", researcher_nodes[0])

        for i in range(len(researcher_nodes) - 1):
            workflow.add_edge(researcher_nodes[i], researcher_nodes[i + 1])

        workflow.add_edge(researcher_nodes[-1], "summarizer")

        logger.info(f"✅ Streaming graph configured with sequential execution: {' → '.join(researcher_nodes)}")

    workflow.add_conditional_edges(
        "summarizer",
        should_continue,
        {researcher_nodes[0]: researcher_nodes[0], "end": END}
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
    workflow.add_node("researcher", arxiv_researcher_node)
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

def create_map_reduce_graph(
    with_checkpointer: bool = True
):
    """
    Creates and compiles a graph with map-reduce pattern for multi-query research.

    The clarifier generates multiple refined queries (controlled by num_queries in state),
    then maps them to parallel ArXiv searches, and reduces the results with a summarizer.

    Args:
        with_checkpointer: Enable state persistence

    Flow:
        clarifier → [arxiv_1, arxiv_2, ...] → summarizer

    Example:
        ```python
        graph = create_map_reduce_graph()
        result = graph.invoke({
            "query": "quantum computing applications",
            "num_queries": 3,  # Generate 3 different search queries (optional, defaults to 1)
            "max_papers": 3     # Each query fetches 3 papers
        })
        ```
    """

    # Define a routing function that creates Send objects for each query
    def continue_to_researchers(state: InternalState):
        """
        Map function: sends each refined query to a separate researcher invocation.
        Uses LangGraph's Send API to dynamically fan out to multiple nodes.
        """
        refined_queries = state.get("refined_queries", [])

        if not refined_queries:
            # Fallback to single query mode
            refined_queries = [state.get("refined_query", state["query"])]

        # Create a Send object for each query
        # Each Send will invoke the researcher node with a modified state
        return [
            Send(
                "arxiv_researcher",
                {
                    **state,
                    "refined_query": query,
                    "iteration": 0
                }
            )
            for query in refined_queries
        ]

    workflow = StateGraph(
        InternalState,
        input=InputState,
        output=OutputState,
    )

    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("arxiv_researcher", arxiv_researcher_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.set_entry_point("clarifier")

    # Use conditional edges with Send for dynamic fan-out (map)
    workflow.add_conditional_edges(
        "clarifier",
        continue_to_researchers,
        ["arxiv_researcher"]
    )

    # All researcher instances converge to summarizer (reduce)
    workflow.add_edge("arxiv_researcher", "summarizer")
    workflow.add_edge("summarizer", END)

    logger.info(f"✅ Map-reduce graph configured")

    if with_checkpointer:
        import sqlite3
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory = SqliteSaver(conn)
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()

async def create_map_reduce_streaming_graph(
    with_checkpointer: bool = True
):
    """
    Creates and compiles a map-reduce graph with streaming support.

    The clarifier generates multiple refined queries (controlled by num_queries in state),
    then maps them to parallel ArXiv searches with streaming, and reduces results.

    Args:
        with_checkpointer: Enable async state persistence

    Returns:
        Compiled async graph with streaming capabilities

    Example:
        ```python
        graph = await create_map_reduce_streaming_graph()
        async for event in graph.astream({
            "query": "quantum computing applications",
            "num_queries": 3  # Generate 3 different search queries (optional, defaults to 1)
        }):
            print(event)
        ```
    """

    # Define a routing function for map phase
    def continue_to_researchers(state: InternalState):
        """Map function: sends each refined query to a separate researcher."""
        refined_queries = state.get("refined_queries", [])

        if not refined_queries:
            refined_queries = [state.get("refined_query", state["query"])]

        return [
            Send(
                "arxiv_researcher",
                {
                    **state,
                    "refined_query": query,
                    "iteration": 0
                }
            )
            for query in refined_queries
        ]

    workflow = StateGraph(
        InternalState,
        input=InputState,
        output=OutputState,
    )

    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("arxiv_researcher", arxiv_researcher_node_streaming)
    workflow.add_node("summarizer", summarizer_node_streaming)

    workflow.set_entry_point("clarifier")

    workflow.add_conditional_edges(
        "clarifier",
        continue_to_researchers,
        ["arxiv_researcher"]
    )

    workflow.add_edge("arxiv_researcher", "summarizer")
    workflow.add_edge("summarizer", END)

    logger.info(f"✅ Map-reduce streaming graph configured")

    if with_checkpointer:
        import aiosqlite
        conn = await aiosqlite.connect(":memory:")
        memory = AsyncSqliteSaver(conn)
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


# Default graph instance for LangGraph Studio
# graph = create_graph(with_checkpointer=False)
# graph = create_graph_with_approval(with_checkpointer=False)
# graph = create_graph(with_checkpointer=False, tools=["arxiv", "wikipedia"], mode="sequential")
# graph = create_graph(with_checkpointer=False, tools=["arxiv", "wikipedia"], mode="parallel")
graph = create_map_reduce_graph(with_checkpointer=False)