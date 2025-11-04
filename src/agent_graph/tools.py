"""ArXiv search tool."""
import arxiv
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor
import threading

# Create a thread pool executor that will be reused
_executor = None
_executor_lock = threading.Lock()


def _get_executor():
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=4)
    return _executor


def _fetch_arxiv_results(query: str, max_results: int) -> list[dict]:
    """
    Helper function to fetch ArXiv results synchronously.
    This is run in a separate thread to avoid blocking the event loop.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        papers.append({
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
        })

    return papers


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
    # Run the blocking operation in a thread pool to avoid blocking the event loop
    executor = _get_executor()
    future = executor.submit(_fetch_arxiv_results, query, max_results)
    return future.result()