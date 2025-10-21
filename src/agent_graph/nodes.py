"""Graph nodes: clarifier, researcher, and summarizer."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent_graph.state import AgentState
from agent_graph.tools import search_arxiv

import logging

# Configure the logger (optional but recommended)
logging.basicConfig(
    level=logging.INFO,                     # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# Create a logger instance
logger = logging.getLogger(__name__)

def clarifier_node(state: AgentState) -> AgentState:
    """Clarifier node: refines and optimizes the user query."""
    # Get config with defaults
    config = state.get("config", {})
    original_query = state["query"]
    
    llm = ChatOpenAI(
        model=config.get("llm_model", "gpt-4o-mini"),
        temperature=config.get("llm_temperature", 0)
    )
    
    print(f"🎯 Clarifying query: '{original_query}'")
    
    messages = [
        SystemMessage(content="""You are a research assistant specialized in query refinement.
        Your task is to transform user questions into optimal ArXiv search queries.
        
        Guidelines:
        - Identify key scientific concepts
        - Use technical terminology
        - Keep it concise (5-10 words max)
        - Focus on the core topic
        
        Return ONLY the refined query, nothing else."""),
        HumanMessage(content=f"Original question: {original_query}", name="User")
    ]
    
    response = llm.invoke(messages)
    refined_query = response.content.strip()
    
    print(f"✨ Refined query: '{refined_query}'")
    
    return {
        **state,
        "refined_query": refined_query,
        "messages": [
            HumanMessage(content=original_query, name="User"),
            AIMessage(content=f"Refined query: {refined_query}", name="Clarifier")
        ]
    }


def researcher_node(state: AgentState) -> AgentState:
    """Researcher node: searches ArXiv for relevant papers."""
    config = state.get("config", {})
    max_papers = config.get("max_papers", 5)
    
    query = state["refined_query"]
    iteration = state.get("iteration", 0)
    
    print(f"🔍 Searching ArXiv: '{query}' (iteration {iteration})")
    
    papers = search_arxiv.invoke({"query": query, "max_results": max_papers})
    
    print(f"📚 Found {len(papers)} papers")
    
    return {
        **state,
        "papers": papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=f"Found {len(papers)} papers on ArXiv for query: {query}",
                name="Researcher"
            )
        ]
    }


def summarizer_node(state: AgentState) -> AgentState:
    """Summarizer node: analyzes papers and generates a concise summary."""
    config = state.get("config", {})
    max_iterations = config.get("max_iterations", 2)
    
    llm = ChatOpenAI(
        model=config.get("llm_model", "gpt-4o-mini"),
        temperature=config.get("llm_temperature", 0)
    )
    
    papers = state["papers"]
    query = state["query"]
    iteration = state["iteration"]
    
    if len(papers) < 3 and iteration < max_iterations:
        print(f"⚠️  Only {len(papers)} papers found, will retry search...")
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
    
    print(f"📝 Synthesizing {len(papers)} papers...")
    
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
    
    print("✅ Summary generated")
    
    return {
        **state, 
        "summary": summary,
        "messages": [
            AIMessage(content=summary, name="Summarizer")
        ]
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge: decides whether to loop back or end.
    
    This function is called after the summarizer node to determine
    the next step in the graph.
    
    Returns:
        - "researcher": if we need more papers (loops back)
        - "end": if we're done (exits the graph)
    """
    config = state.get("config", {})
    max_iterations = config.get("max_iterations", 2)
    
    if state.get("summary") == "NEED_MORE_PAPERS" and state["iteration"] < max_iterations:
        return "researcher"
    return "end"