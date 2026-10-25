"""EEG-Agent Core public API. Importing it does not start a model or web UI.

EEGTools requires scientific dependencies; AgentGraphs is only loaded when used.
"""
__version__ = "0.4.2"
__all__ = ["EEGSession", "STRATEGIES", "EEGTools", "SUPPORTED_RAW", "AgentGraphs", "build_agent", "channel_config_path", "extract_handcrafted_features", "DEFAULT_BANDS", "FeatureMatrix", "SessionStore", "compute_connectivity", "assess_windows", "review_preprocessing", "preprocess_dataset", "paired_quality_report", "validate_output_metadata", "output_metadata_schema_path"]


def __getattr__(name):
    if name in {"paired_quality_report", "validate_output_metadata"}:
        from . import quality_reports
        return getattr(quality_reports, name)
    if name == "compute_connectivity":
        from .connectivity import compute_connectivity
        return compute_connectivity
    if name == "assess_windows":
        from .qc import assess_windows
        return assess_windows
    if name == "review_preprocessing":
        from .science import review_preprocessing
        return review_preprocessing
    if name == "preprocess_dataset":
        from .batch import preprocess_dataset
        return preprocess_dataset
    if name == "SessionStore":
        from .persistence import SessionStore
        return SessionStore
    if name in {"EEGSession", "STRATEGIES"}:
        from . import session
        return getattr(session, name)
    if name in {"EEGTools", "SUPPORTED_RAW"}:
        from . import tools
        return getattr(tools, name)
    if name in {"extract_handcrafted_features", "DEFAULT_BANDS", "FeatureMatrix"}:
        from . import features
        return getattr(features, name)
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


def output_metadata_schema_path():
    """Bundled JSON Schema for the versioned EEG-Agent recording sidecar."""
    from importlib.resources import files
    from pathlib import Path
    return Path(str(files(__name__).joinpath("resources", "output_metadata.schema.json")))
