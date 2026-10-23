"""EEG-Agent Core public API. Importing it does not start a model or web UI.

EEGTools requires scientific dependencies; AgentGraphs is only loaded when used.
"""
__version__ = "0.3.0"
__all__ = ["EEGSession", "STRATEGIES", "EEGTools", "SUPPORTED_RAW", "AgentGraphs", "build_agent", "channel_config_path"]


def __getattr__(name):
    if name in {"EEGSession", "STRATEGIES"}:
        from . import session
        return getattr(session, name)
    if name in {"EEGTools", "SUPPORTED_RAW"}:
        from . import tools
        return getattr(tools, name)
    if name in {"AgentGraphs", "build_agent"}:
        from . import adapter
        return getattr(adapter, name)
    raise AttributeError(name)


def channel_config_path(name: str):
    """Return a bundled channel layout JSON path, e.g. ``standard_26``.

    The caller is responsible for ensuring the requested montage is suitable
    for the measured electrodes. No missing channel is silently synthesized.
    """
    from importlib.resources import files
    from pathlib import Path
    allowed = {"standard_26", "standard_64", "hydrocel_to_26", "hydrocel_to_64"}
    if name not in allowed:
        raise ValueError(f"Unknown channel config: {name}")
    return Path(str(files(__name__).joinpath("resources", name + ".json")))
