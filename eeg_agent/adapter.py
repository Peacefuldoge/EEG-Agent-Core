"""Optional LangGraph adapter: plug in a user-provided model and EEGTools.

No vLLM deployment, Chainlit application, or terminal UI is imported here.
"""
from __future__ import annotations


def build_agent(*, strategy: str, model, tools, checkpointer=None,
                max_steps: int = 12, max_replans: int = 1):
    """Return a compiled LangGraph graph for the selected strategy.

    ``model`` may be a LangChain chat model (e.g. user's ChatOpenAI) or
    an object already implementing plan(), react(), and summarize().
    """
    from .model import QwenModel
    from .strategies import AgentGraphs
    adapter = model if all(callable(getattr(model, n, None))
                           for n in ("plan", "react", "summarize")) else QwenModel(chat=model)
    return AgentGraphs(adapter, tools, max_steps=max_steps,
                       max_replans=max_replans).build(strategy, checkpointer)
