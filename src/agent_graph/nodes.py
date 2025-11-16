"""Graph nodes: clarifier, researcher, and summarizer."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt

from agent_graph.state import InputState, OutputState, PrivateState, InternalState
from agent_graph.tools import search_arxiv, search_arxiv_streaming


import logging

# Configure the logger (optional but recommended)
logging.basicConfig(
    level=logging.INFO,                     # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# Create a logger instance
logger = logging.getLogger(__name__)

# ============================================================================
# SYNC (NON-STREAMING) NODES
# ============================================================================


def clarifier_node(state: InternalState) -> PrivateState:
    """Clarifier node: refines and optimizes the user query."""
    
    original_query = state["query"]
    conversation_history = state.get("messages", [])

    llm = ChatOpenAI(
        model=state.get("llm_model", "gpt-4o-mini"),
        temperature=state.get("llm_temperature", 0)
    )

    logging.info(f"Clarifying query: '{original_query}'")

    # Build messages with conversation history
    system_prompt = """You are a research assistant specialized in query refinement.
Your task is to transform user questions into optimal ArXiv search queries.

Guidelines:
- Consider the conversation history to understand context
- If the user asks a follow-up question (e.g., "tell me more about X", "what about Y?"), use the previous context to refine the query
- Identify key scientific concepts
- Use technical terminology
- Keep it concise (5-10 words max)
- Focus on the core topic

Return ONLY the refined query, nothing else."""

    messages = [SystemMessage(content=system_prompt)]

    # Add conversation history for context
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current query
    messages.append(HumanMessage(content=f"Original question: {original_query}", name="User"))

    response = llm.invoke(messages)
    refined_query = response.content.strip()

    logging.info(f"Refined query: '{refined_query}'")

    # Return PrivateState fields
    return {
        "refined_query": refined_query,
        "iteration": 0,
        "messages": [
            HumanMessage(content=original_query, name="User"),
            AIMessage(content=f"Refined query: {refined_query}", name="Clarifier")
        ]
    }


def researcher_node(state: InternalState) -> OutputState:
    """Researcher node: searches ArXiv for relevant papers and scores their relevance."""
   
    max_papers = state.get("max_papers")

    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"Searching ArXiv: '{query}' (iteration {iteration})")
    papers = search_arxiv.invoke({"query": query, "max_results": max_papers})
    logging.info(f"Found {len(papers)} papers")

    llm_model = state.get("llm_model", "gpt-4o-mini")
    llm_temperature = state.get("llm_temperature", 0)

    # Score each paper's relevance using LLM
    llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    logging.info(f"Scoring paper relevance...")

    scored_papers = []
    for paper in papers:
        # Create prompt for scoring
        score_messages = [
            SystemMessage(content="""You are a research relevance evaluator.
            Score how relevant a paper is to the user's query on a scale from 1 to 100.

            Consider:
            - Direct relevance to the query topic
            - Quality and depth of content based on the abstract
            - Potential usefulness for answering the query

            Respond with ONLY a number between 1 and 100, nothing else."""),
            HumanMessage(content=f"""User Query: {original_query}

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:3])}
Abstract: {paper['summary'][:500]}

Relevance Score (1-100):""", name="User")
        ]

        response = llm.invoke(score_messages)
        relevance_score = int(response.content.strip())
        paper['relevance_score'] = max(1, min(100, relevance_score))
        scored_papers.append(paper)
        logging.info(f"  📄 {paper['title'][:60]}... - Score: {relevance_score}")

    # Return OutputState fields
    return {
        "papers": scored_papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=f"Found {len(scored_papers)} papers on ArXiv for query: {query}",
                name="Researcher"
            )
        ]
    }


def summarizer_node(state: InternalState) -> OutputState:
    """Summarizer node: analyzes papers and generates a concise summary."""

    max_iterations = state.get("max_iterations", 2)
    llm_model = state.get("llm_model", "gpt-4o-mini")
    llm_temperature = state.get("llm_temperature", 0)
    
    llm = ChatOpenAI(model=llm_model,temperature=llm_temperature)
    
    papers = state["papers"]
    query = state["query"]
    iteration = state["iteration"]
    
    if len(papers) < 3 and iteration < max_iterations:
        logging.info(f"⚠️  Only {len(papers)} papers found, will retry search...")
        return {
            **state, 
            "summary": "NEED_MORE_PAPERS",
            "messages": [
                AIMessage(
                    content=f"Insufficient papers ({len(papers)}), retrying search...",
                    name="Summarizer"
                )
            ]
        }
    
    logging.info(f"📝 Synthesizing {len(papers)} papers...")
    
    papers_context = "\n\n".join([
        f"**Paper {i+1}:** {p['title']}\n"
        f"Authors: {', '.join(p['authors'][:3])}\n"
        f"Published: {p['published']}\n"
        f"Abstract: {p['summary'][:400]}...\n"
        f"URL: {p['url']}"
        for i, p in enumerate(papers)
    ])
    
    messages = [
        SystemMessage(content="""You are an expert scientific assistant.
        Create a concise summary with:
        - 3-5 bullet points highlighting main insights of 3-5 sentences
        - Each bullet point must reference source papers [Paper X]
        - Clear and accessible style
        - End with a "References" section listing all papers
        
        Format:
        ## Résumé
        • Point 1 [Paper 1, Paper 2]
        • Point 2 [Paper 3]
        ...
        
        ## Références
        [Paper 1] Title - Authors (Year) - URL
        ..."""),
        HumanMessage(content=f"""Original question: {query}
        
Papers found:
{papers_context}

Generate a structured summary.""", name="User")
    ]
    
    response = llm.invoke(messages)
    summary = response.content
    
    logging.info("✅ Summary generated")
    
    # Return OutputState fields
    return {
        "summary": summary,
        "messages": [
            AIMessage(content=summary, name="Summarizer")
        ]
    }


def should_continue(state: InternalState) -> str:
    """
    Conditional edge: decides whether to loop back or end.

    This function is called after the summarizer node to determine
    the next step in the graph.

    Returns:
        - "researcher": if we need more papers (loops back)
        - "end": if we're done (exits the graph)
    """
    max_iterations = state.get("max_iterations", 2)

    if state.get("summary") == "NEED_MORE_PAPERS" and state["iteration"] < max_iterations:
        return "researcher"
    return "end"


# ============================================================================
# ASYNC STREAMING NODES
# ============================================================================

async def researcher_node_streaming(state: InternalState) -> OutputState:
    """
    Async researcher node with streaming support.

    This version uses the streaming ArXiv tool and streams token-level
    updates during relevance scoring.
    """
    max_papers = state.get("max_papers", 5)
    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"🔍 Searching ArXiv: '{query}' (iteration {iteration})")

    # Use streaming version of search
    papers = await search_arxiv_streaming.ainvoke({"query": query, "max_results": max_papers})
    logging.info(f"Found {len(papers)} papers")

    llm_model = state.get("llm_model", "gpt-4o-mini")
    llm_temperature = state.get("llm_temperature", 0)

    # Score each paper's relevance using LLM with streaming
    llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    logging.info(f"Scoring paper relevance...")

    scored_papers = []
    for i, paper in enumerate(papers, 1):
        # Create prompt for scoring
        score_messages = [
            SystemMessage(content="""You are a research relevance evaluator with expertise in academic paper assessment.

Score how relevant this paper is to the user's query on a scale from 1 to 100.

SCORING CRITERIA:

**High Relevance (80-100):**
- Directly addresses the core topic/question in the query
- Abstract demonstrates deep technical/conceptual alignment
- Contains specific methods, results, or insights that answer the query
- Published in reputable venue with clear contributions

**Moderate Relevance (50-79):**
- Partially addresses the query or covers related subtopics
- Abstract shows tangential connection or addresses one aspect of the query
- Provides useful background or context but not a direct answer
- May require inference to connect to the query

**Low Relevance (20-49):**
- Mentions query keywords but focuses on different aspects
- Abstract shows weak conceptual overlap
- Provides minimal useful information for the query
- May be from a related field but different application domain

**Irrelevant (1-19):**
- Shares only superficial keyword overlap
- Abstract shows the paper addresses a different problem entirely
- Would not help answer the query in any meaningful way

EVALUATION FACTORS:
1. **Topic alignment**: Does the paper's main focus match the query intent?
2. **Methodological relevance**: Are the techniques/approaches applicable to the query?
3. **Depth of coverage**: Does the abstract suggest comprehensive treatment of relevant concepts?
4. **Recency and impact**: Is the paper recent/seminal enough to provide current insights?
5. **Specificity**: Does it address the specific aspect mentioned in the query or just general concepts?

Respond with ONLY a number between 1 and 100. No explanation, no text, just the number."""),
            HumanMessage(content=f"""User Query: {original_query}

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:3])}
Abstract: {paper['summary'][:500]}

Relevance Score (1-100):""", name="User")
        ]

        response = await llm.ainvoke(score_messages)
        relevance_score = int(response.content.strip())
        paper['relevance_score'] = max(1, min(100, relevance_score))
        scored_papers.append(paper)
        logging.info(f"  📄 {paper['title'][:60]}... - Score: {relevance_score}")

    # Return OutputState fields
    return {
        "papers": scored_papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=f"Found {len(scored_papers)} papers on ArXiv for query: {query}",
                name="Researcher"
            )
        ]
    }


async def summarizer_node_streaming(state: InternalState) -> OutputState:
    """
    Async summarizer node with token-level streaming.

    This version streams the summary generation token by token,
    providing real-time feedback to users.
    """
    max_iterations = state.get("max_iterations", 2)

    llm_model = state.get("llm_model", "gpt-4o-mini")
    llm_temperature = state.get("llm_temperature", 0)

    llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    papers = state["papers"]
    query = state["query"]
    iteration = state["iteration"]

    if len(papers) < 3 and iteration < max_iterations:
        logging.info(f"⚠️  Only {len(papers)} papers found, will retry search...")
        return {
            **state,
            "summary": "NEED_MORE_PAPERS",
            "messages": [
                AIMessage(
                    content=f"Insufficient papers ({len(papers)}), retrying search...",
                    name="Summarizer"
                )
            ]
        }

    logging.info(f"📝 Synthesizing {len(papers)} papers with streaming...")

    papers_context = "\n\n".join([
        f"**Paper {i+1}:** {p['title']}\n"
        f"Authors: {', '.join(p['authors'][:3])}\n"
        f"Published: {p['published']}\n"
        f"Abstract: {p['summary'][:400]}...\n"
        f"URL: {p['url']}"
        for i, p in enumerate(papers)
    ])

    messages = [
        SystemMessage(content="""You are an expert scientific assistant.
        Create a concise summary with:
        - 3-5 bullet points highlighting main insights of 3-5 sentences
        - Each bullet point must reference source papers [Paper X]
        - Clear and accessible style
        - End with a "References" section listing all papers

        Format:
        ## Résumé
        • Point 1 [Paper 1, Paper 2]
        • Point 2 [Paper 3]
        ...

        ## Références
        [Paper 1] Title - Authors (Year) - URL
        ..."""),
        HumanMessage(content=f"""Original question: {query}

Papers found:
{papers_context}

Generate a structured summary.""", name="User")
    ]

    # Stream the response token by token
    summary_chunks = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            summary_chunks.append(chunk.content)

    summary = "".join(summary_chunks)

    logging.info("✅ Summary generated")

    # Return OutputState fields
    return {
        "summary": summary,
        "messages": [
            AIMessage(content=summary, name="Summarizer")
        ]
    }


def approver_node(state: InternalState) -> dict:
    """
    A node that explicitly requests human approval before proceeding.
    
    This demonstrates the node-level interrupt pattern where the node
    itself decides to pause and request input.
    """
    
    refined_query = state.get("refined_query", "")
    
    # This is where the magic happens - we interrupt and ask for a decision
    approval_response = interrupt({
        "type": "approval_request",
        "query": refined_query,
        "message": (
            f"The clarifier suggests this search query:\n\n"
            f"   '{refined_query}'\n\n"
            f"What would you like to do?\n"
            f"  - Type 'approve' to continue\n"
            f"  - Type 'edit' to modify the query\n"
            f"  - Type 'cancel' to stop"
        )
    })
    
    # Execution pauses at the interrupt() call above.
    # When you resume with Command(resume=...), that value appears here.
    
    action = approval_response.get("action", "cancel")
    
    if action == "cancel":
        return {
            "approved": False,
            "messages": [AIMessage(
                content="❌ Search cancelled by user",
                name="Approval"
            )]
        }
    
    if action == "edit":
        new_query = approval_response.get("new_query", refined_query)
        return {
            "refined_query": new_query,
            "approved": True,
            "messages": [AIMessage(
                content=f"✏️ Query updated to: {new_query}",
                name="Approval"
            )]
        }
    
    # If we get here, user approved
    return {
        "approved": True,
        "messages": [AIMessage(
            content="✅ Query approved - proceeding to search",
            name="Approval"
        )]
    }

def route_after_approval(state: InternalState) -> str:
    if state.get("approved", False):
        return "researcher"  # Approved - continue to search
    return "end"  # Cancelled - stop here

