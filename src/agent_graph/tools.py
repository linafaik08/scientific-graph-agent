"""ArXiv search tool."""
import arxiv
from langchain_core.tools import tool


@tool
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for scientific papers on ArXiv.
    
    This tool queries the ArXiv API and returns relevant papers
    with their metadata (title, authors, abstract, URL).
    
    Args:
        query: Search keywords (e.g., "transformer attention mechanism")
        max_results: Maximum number of papers to return (default: 5)
    
    Returns:
        List of papers with structured metadata
    """
    # Create ArXiv search with relevance sorting
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
        })
    
    return papers