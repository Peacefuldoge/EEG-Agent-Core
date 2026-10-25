"""Shared deterministic scientific/QC policy for all three agent strategies.

A tool request that needs human review must NOT become an implicit approval.
The planner may replan towards non-destructive inspection tools instead.
"""
from __future__ import annotations
from dataclasses import dataclass
from .science import review_preprocessing

REVIEW_GATED_TOOLS=frozenset({
    'preprocess_eeg','batch_preprocess','extract_features',
    'analyze_spectrum','analyze_connectivity','visualize_eeg',
})

@dataclass(frozen=True)
class Decision:
    action:str       # allow, review, reject
    reason:str=''


def evaluate_tool(name:str, arguments:dict, qc_summary:dict | None) -> Decision:
    """Reject invalid settings; otherwise require approval for flagged/unknown QC.

    Tool schemas and path permissions are independently validated by EEGTools.
    This policy does not attempt to diagnose EEG or automatically repair samples.
    """
    if name=='preprocess_eeg':
        report=review_preprocessing(arguments,input_sfreq=arguments.get('input_sfreq'))
        if report.errors:return Decision('reject','; '.join(report.errors))
    if name=='batch_preprocess':
        report=review_preprocessing(arguments.get('preprocessing') or {})
        if report.errors:return Decision('reject','; '.join(report.errors))
    if name not in REVIEW_GATED_TOOLS:return Decision('allow')
    summary=qc_summary or {}
    status=summary.get('status','unavailable')
    if status in ('requires_review','unavailable','invalid','failed'):
        return Decision('review',
            f'QC status {status} requires explicit researcher approval before {name}. '
            'Inspect the source, address QC findings, or request approval.')
    if status!='pass':
        return Decision('review',f'Unknown QC status {status}; researcher approval required.')
    return Decision('allow')
