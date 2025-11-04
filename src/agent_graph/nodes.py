"""Graph nodes: clarifier, researcher, and summarizer."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent_graph.state import InputState, OutputState, PrivateState, InternalState
from agent_graph.tools import search_arxiv

import logging

# Configure the logger (optional but recommended)
logging.basicConfig(
    level=logging.INFO,                     # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# Create a logger instance
logger = logging.getLogger(__name__)

def clarifier_node(state: InternalState) -> PrivateState:
    """Clarifier node: refines and optimizes the user query."""
    
    original_query = state["query"]
    conversation_history = state.get("messages", [])

    llm = ChatOpenAI(
        model=state.get("llm_model", "gpt-4o-mini"),
        temperature=state.get("llm_temperature", 0)
    )

    logging.info(f"🎯 Clarifying query: '{original_query}'")

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

    logging.info(f"✨ Refined query: '{refined_query}'")

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
    config = state.get("config", {})
    max_papers = config.get("max_papers", 5)

    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"Searching ArXiv: '{query}' (iteration {iteration})")
    papers = search_arxiv.invoke({"query": query, "max_results": max_papers})
    logging.info(f"Found {len(papers)} papers")

    max_papers = state.get("max_papers", 5)
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
    config = state.get("config", {})
    max_iterations = config.get("max_iterations", 2)

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
    iteration = state.get("iteration", 0)
    
    if state.get("summary") == "NEED_MORE_PAPERS" and state["iteration"] < max_iterations:
        return "researcher"
    return "end"