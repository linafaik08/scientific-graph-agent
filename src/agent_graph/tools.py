"""ArXiv search tool with streaming support."""
import arxiv
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio

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
    # Create a custom client with more conservative rate limiting
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=1.0,  # Increased from default 3.0
        num_retries=1      # Increased retries
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
        })

    return papers


async def _fetch_arxiv_results_async(query: str, max_results: int, writer=None) -> list[dict]:
    """
    Async helper function to fetch ArXiv results with progress updates.

    Args:
        query: Search keywords
        max_results: Maximum number of papers to return
        writer: Stream writer for custom progress updates (optional)

    Returns:
        List of papers with structured metadata
    """

    papers = []
    if writer and callable(writer):
        result = writer(f"Starting ArXiv search for: '{query}'")

    loop = asyncio.get_running_loop()

    def _search_with_rate_limit():
        """Helper to run search with rate limiting."""

        # Create a custom client with more conservative rate limiting
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=1.0,  # Increased from default 3.0
            num_retries=1       # Increased retries
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        return list(client.results(search))

    results = await loop.run_in_executor(_get_executor(), _search_with_rate_limit)

    for i, result in enumerate(results, 1):
        paper = {
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
        }
        papers.append(paper)

        if writer and callable(writer):
            # Emit progress update
            writer(f"Found paper {i}/{len(results)}: {result.title[:60]}...")

    if writer and callable(writer):
        result = writer(f"✅ Search complete: {len(papers)} papers found")

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


@tool
async def search_arxiv_streaming(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for scientific papers on ArXiv with streaming progress updates.

    This async tool queries the ArXiv API and emits progress updates
    as papers are found. Use this within LangGraph to get real-time
    search progress.

    Args:
        query: Search keywords (e.g., "transformer attention mechanism")
        max_results: Maximum number of papers to return (default: 5)

    Returns:
        List of papers with structured metadata
    """

    try:
        writer = get_stream_writer()
        # Check if writer is actually callable
        if writer is None or not callable(writer):
            writer = None
    except Exception:
        # If not in LangGraph context, fall back to no streaming
        writer = None

    return await _fetch_arxiv_results_async(query, max_results, writer)