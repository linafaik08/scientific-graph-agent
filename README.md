# Scientific Graph Agent: A Simple Agent Graph for Paper Exploration
**A Practical Guide to Building Multi-Node Agent Systems with LangGraph**

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: October 2025  
- Last update: November 2025

## Objective

This repository demonstrates **a minimal yet powerful agent graph architecture** using LangGraph. It showcases how to build autonomous research assistants that clarify queries, search scientific literature, and synthesize findings—all with built-in memory, retry logic, and full observability through LangSmith Studio.

For a deeper dive into the concepts and implementation details, check out the full article on [The AI Practitioner](https://aipractitioner.substack.com/).

## Project Description

Traditional linear pipelines fall short when building intelligent agents that need to adapt based on results, retry when data is insufficient, or maintain conversation context. This project implements a **3-node agent graph** with conditional edges and persistent memory, demonstrating core patterns for production-grade agent systems.

### System Architecture

The scientific paper explorer consists of three specialized nodes:

1. **Clarifier Node**: Autonomously refines user questions into precise academic search queries
2. **Researcher Node**: Searches ArXiv API for relevant papers (with configurable result count)
3. **Summarizer Node**: Analyzes papers and generates bullet-point summaries with references

The graph includes **intelligent retry logic**: if fewer than 3 papers are found, it automatically loops back to search again (up to a configurable max iteration limit).

### Code Structure

```
notebooks/
├── demo_basic.ipynb           # Memory, checkpoints & conversation flow
├── demo_advanced.ipynb        # Streaming, interrupts & time travel
└── demo_multi_tools.ipynb     # Multi-tool (ArXiv+Wikipedia) & map-reduce

src/agent_graph/
├── state.py                   # State management with reducers
├── tools.py                   # ArXiv & Wikipedia search (sync + streaming)
├── nodes.py                   # Clarifier, Researcher, Summarizer, Approver
└── graph.py                   # Graph builder with interrupt support
```

## How to Use This Repository?

### Requirements

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

Main libraries:
```
# LangGraph and AI framework
langgraph>=0.2.53
langgraph-prebuilt>=0.0.1
langgraph-sdk>=0.1.38
langgraph-checkpoint-sqlite>=2.0.4
langgraph-cli[inmem]>=0.1.58
langsmith>=0.1.147

# LangChain ecosystem
langchain-community>=0.3.8
langchain-core>=0.3.21
langchain-openai>=0.2.9

# Scientific paper search
arxiv>=2.1.3

# Utilities
python-dotenv>=1.0.1
```

### Installation

1. **Install uv** (if not already installed)

2. **Clone the repository**:
```bash
git clone <your-repo-url>
cd scientific-graph-agent
```

3. **Install dependencies with uv**:
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install the package
uv pip install -e ".[dev]"
```

### Setup

1. **Create a `.env` file** based on `.env.example`:
```bash
cp .env.example .env
```

2. **Add your API keys** to `.env`:
```
OPENAI_API_KEY=sk-your-key-here

# Optional: For LangSmith tracing and Studio visualization
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_key_here
LANGCHAIN_PROJECT=scientific-graph-agent
```

3. **Initialize LangGraph Studio** (optional):
```bash
langgraph dev
```

### Running the Project

1. **Start with the demo**: Run the interactive demo script to see the complete agent system in action:
```notebooks/demo.ipynb```

2. **Explore observability**: 
   - View execution traces in LangSmith at https://smith.langchain.com
   - Visualize the graph in real-time with LangGraph Studio (https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)
   - Inspect state transitions and conditional branching

3. **Customize the graph**: Modify parameters when creating the graph:
```python
from agent_graph import create_graph

# Configuration
LLM_MODEL = "gpt-4o-mini"
config = {"configurable": {"thread_id": "demo-thread-1"}}

# Create graph (no checkpointer for Studio, True for local)
graph = create_graph(with_checkpointer=True)

# Define your question
question = "What are the key innovations in transformer architectures?"

# Prepare initial state
initial_state = {
    "query": question,
    "config": {
        "llm_model": LLM_MODEL,
        "llm_temperature": 0,
        "max_papers": 5,
        "max_iterations": 2,
    }
}

# Invoke the graph
result = graph.invoke(initial_state, config=config)

# Display results
print(f"Original query: {result['query']}")
print(f"Refined query: {result['refined_query']}")
print(f"Papers found: {len(result['papers'])}")
print(f"Iterations: {result['iteration']}")
print(f"\n{result['summary']}")
```

### Key Features Demonstrated

- **Multi-Node Architecture**: Clarifier → Researcher → Approver → Summarizer with conditional routing
- **Multiple Execution Modes**: Sequential queries or map-reduce parallel search
- **Multi-Source Research**: ArXiv papers + Wikipedia articles with streaming support
- **Human-in-the-Loop**: Interrupt points for approval before continuing
- **Intelligent Retry Logic**: Auto-loop when results insufficient (configurable thresholds)
- **Persistent Memory**: SQLite checkpointer with message summarization
- **State Reducers**: Smart paper deduplication and conversation history management
- **Time Travel Debugging**: Replay from any checkpoint state
- **Full Observability**: LangSmith tracing + LangGraph Studio visualization

## Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangSmith Studio Guide**: https://docs.smith.langchain.com/
- **ArXiv API**: https://info.arxiv.org/help/api/index.html
- **uv Package Manager**: https://github.com/astral-sh/uv

## License

MIT License - Free to use for learning and production projects.

---

*Built with ❤️ to demonstrate clean agent architecture with LangGraph*