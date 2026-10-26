from __future__ import annotations
from typing import Any, Literal
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator, ValidationInfo

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class InspectArgs(StrictModel):
    source: str = Field(description='EEG file path or $input/$latest_npz reference')
    input_sfreq: float | None = Field(default=None, gt=0)
    input_channels: list[str] | None = None
    input_unit: Literal['V', 'mV', 'uV'] | None = None

class PreprocessArgs(StrictModel):
    source: str
    target_sfreq: float = Field(default=250.0, gt=50, le=4096)
    window_seconds: float = Field(default=1.0, gt=0, le=60)
    stride_seconds: float | None = Field(default=None, gt=0, le=60)
    target_count: int | None = Field(default=None, ge=1, le=256)
    target_channels: list[str] | None = None
    input_sfreq: float | None = Field(default=None, gt=0)
    input_channels: list[str] | None = None
    input_unit: Literal['V', 'mV', 'uV'] | None = None
    bandpass_low: float = Field(default=0.5, ge=0)
    bandpass_high: float = Field(default=45.0, gt=0)
    notch_freqs: list[float] = Field(default_factory=lambda: [50.,60.])
    use_ica: bool = True
    ignore_annotations: bool = False
    events_tsv: str | None = None
    interpolate_missing: bool = False
    use_cache: bool = True

    @model_validator(mode='after')
    def valid_combo(self):
        if self.target_count is not None and self.target_channels is not None:
            raise ValueError('Use target_count OR target_channels')
        if self.bandpass_low >= self.bandpass_high or self.bandpass_high * 2 >= self.target_sfreq:
            raise ValueError('Invalid filtering band relative to sampling rate')
        if self.ignore_annotations and self.events_tsv:
            raise ValueError('Cannot use events_tsv with ignore_annotations')
        if any(x <= 0 or x >= self.target_sfreq / 2 for x in self.notch_freqs):
            raise ValueError('notch_freqs must be positive and below target Nyquist')
        return self

class SpectrumArgs(StrictModel):
    source: str = Field(description='Preprocessed NPZ path or $latest_npz')
    max_windows: int = Field(default=32, ge=1, le=128)

class FeatureArgs(StrictModel):
    source: str = Field(description='Preprocessed EEG NPZ path or $latest_npz')
    max_windows: int = Field(default=32, ge=1, le=512)
    # Legacy NPZ without data_unit are assumed to be uV; can explicitly override.
    assume_unit: Literal['uV', 'V', 'mV'] = 'uV'
    bands: dict[str, tuple[float, float]] | None = Field(
        default=None, description='Optional named [low_hz, high_hz] EEG frequency bands')
    total_band: tuple[float, float] = (1.0, 45.0)
    band_de_method: Literal['filtered', 'psd'] = 'filtered'
    welch_seconds: float = Field(default=2.0, gt=0, le=60)
    filter_order: int = Field(default=4, ge=1, le=12)

class ConnectivityArgs(StrictModel):
    source: str
    max_windows: int = Field(default=32, ge=1, le=128)
    method: Literal['pearson','coh','imcoh','plv','pli','wpli'] = 'pearson'
    backend: Literal['scipy','mne'] = 'scipy'
    bands: dict[str,tuple[float,float]] | None = None
    segment_seconds: float = Field(default=1., gt=0, le=10)

class QcArgs(StrictModel):
    source: str
    input_unit: Literal['V', 'mV', 'uV'] | None = None
    max_windows: int = Field(default=32,ge=1,le=256)
    max_raw_seconds: float = Field(default=120.,gt=0,le=1200)
    flat_uv: float = Field(default=.25,gt=0,le=20)
    high_amplitude_uv: float = Field(default=250.,gt=0,le=2000)
    line_ratio_limit: float = Field(default=5.,gt=1,le=100)

class QualityReportArgs(StrictModel):
    source: str = Field(description='Preprocessed EEG NPZ path; QC sidecars generated during preprocessing')

class MetadataArgs(StrictModel):
    source: str = Field(description='Preprocessed EEG NPZ path with standardized output metadata sidecar')

class IntegrityArgs(StrictModel):
    source: str
    max_windows: int = Field(default=64,ge=1,le=512)

class BatchArgs(StrictModel):
    source_dir: str
    preprocessing: dict[str,Any] = Field(default_factory=dict)
    recursive: bool = True
    max_files: int = Field(default=1000,ge=1,le=10000)
    continue_on_error: bool = True

class ProposalArgs(StrictModel):
    source: str
    qc_max_windows: int = Field(default=32,ge=1,le=256)

class VisualizeArgs(StrictModel):
    source: str
    plots: str = Field(default='default', description='default, all, or comma-separated existing plot names')
    max_windows: int = Field(default=32, ge=1, le=128)

BUILTIN_TOOL_NAMES = frozenset({'inspect_eeg','preprocess_eeg','analyze_spectrum',
    'analyze_connectivity','visualize_eeg','extract_features','assess_eeg_quality',
    'propose_preprocessing','validate_eeg','batch_preprocess',
    'compare_eeg_quality','inspect_output_metadata'})

class Step(StrictModel):
    id: str = Field(pattern=r'^(?:E[1-9][0-9]*|P(?:[2-9]|[1-9][0-9]+):E[1-9][0-9]*)$')
    tool: str = Field(min_length=1,max_length=64,pattern=r'^[a-z][a-z0-9_]*$')
    args: dict[str,Any]
    depends_on: list[str] = Field(default_factory=list)

    @field_validator('tool')
    @classmethod
    def check_allowed(cls,value,info:ValidationInfo):
        available=(info.context or {}).get('allowed_tools',BUILTIN_TOOL_NAMES)
        if value not in available:
            raise ValueError('Tool is not registered/allowed: '+value)
        return value

class Plan(StrictModel):
    steps: list[Step] = Field(min_length=1,max_length=12)

    @model_validator(mode='after')
    def validate_dependencies(self):
        names = [s.id for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError('Step IDs must be unique')
        ids=set(names)
        by_id={s.id:set(s.depends_on) for s in self.steps}
        for s in self.steps:
            if s.id in s.depends_on or any(dep not in ids for dep in s.depends_on):
                raise ValueError(f'{s.id} has a self-reference or unknown dependency')
            if len(s.depends_on)!=len(set(s.depends_on)):
                raise ValueError('Duplicate dependencies')
        visited=set(); visiting=set()
        def visit(sid):
            if sid in visiting:raise ValueError(f'Cyclic dependency detected at {sid}')
            if sid in visited:return
            visiting.add(sid)
            for dep in by_id[sid]:visit(dep)
            visiting.remove(sid);visited.add(sid)
        for sid in names:visit(sid)

        # A planned DAG cannot rely on the execution-order-dependent
        # $latest_npz mutable pointer. Bind it to precisely one preceding
        # producer, and validate every local $E*.field data-flow dependency.
        def refs(obj):
            if isinstance(obj,str) and obj.startswith('$'):
                yield obj
            elif isinstance(obj,dict):
                for value in obj.values():yield from refs(value)
            elif isinstance(obj,list):
                for value in obj:yield from refs(value)

        def replace_latest(obj, producer):
            if obj=='$latest_npz':return f'${producer}.output_npz'
            if isinstance(obj,dict):
                return {key:replace_latest(value,producer) for key,value in obj.items()}
            if isinstance(obj,list):return [replace_latest(value,producer) for value in obj]
            return obj

        def ancestors(sid):
            result=set()
            def walk(node):
                for dep in by_id[node]:
                    if dep not in result:
                        result.add(dep);walk(dep)
            walk(sid)
            return result

        for step in self.steps:
            upstream=ancestors(step.id)
            if '$latest_npz' in set(refs(step.args)):
                producers=[other.id for other in self.steps
                           if other.id in upstream and other.tool=='preprocess_eeg']
                if len(producers)!=1:
                    raise ValueError(
                        f'{step.id}: $latest_npz needs exactly one dependency-reachable '
                        'preprocess_eeg producer (found '+str(len(producers))+'); '
                        'use an explicit $E#.output_npz reference and depends_on')
                step.args=replace_latest(step.args,producers[0])
            for ref in refs(step.args):
                # P2:E1 and P1:E1 are cross-plan qualified references, not
                # current-plan dependencies; activation checks their evidence.
                match=re.fullmatch(r'\$((?:P[2-9][0-9]*:)?E[1-9][0-9]*)\.[a-z][a-zA-Z0-9_]*',ref)
                if match and match.group(1) in ids and match.group(1) not in upstream:
                    raise ValueError(
                        f'{step.id}: reference {ref} requires depends_on '
                        f'path to {match.group(1)}')
        return self
