from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class InspectArgs(StrictModel):
    source: str = Field(description='EEG file path or $input/$latest_npz reference')
    input_sfreq: float | None = Field(default=None, gt=0)
    input_channels: list[str] | None = None

class PreprocessArgs(StrictModel):
    source: str
    target_sfreq: float = Field(default=250.0, gt=50, le=4096)
    window_seconds: float = Field(default=1.0, gt=0, le=60)
    stride_seconds: float | None = Field(default=None, gt=0, le=60)
    target_count: int | None = Field(default=None, ge=1, le=256)
    target_channels: list[str] | None = None
    input_sfreq: float | None = Field(default=None, gt=0)
    input_channels: list[str] | None = None
    bandpass_low: float = Field(default=0.5, ge=0)
    bandpass_high: float = Field(default=45.0, gt=0)
    notch_freqs: list[float] = Field(default_factory=lambda: [50.,60.])
    use_ica: bool = True
    ignore_annotations: bool = False
    events_tsv: str | None = None
    interpolate_missing: bool = False

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

class ConnectivityArgs(StrictModel):
    source: str
    max_windows: int = Field(default=32, ge=1, le=128)

class VisualizeArgs(StrictModel):
    source: str
    plots: str = Field(default='default', description='default, all, or comma-separated existing plot names')
    max_windows: int = Field(default=32, ge=1, le=128)

class Step(StrictModel):
    id: str = Field(pattern=r'^E[1-9][0-9]*$')
    tool: Literal['inspect_eeg','preprocess_eeg','analyze_spectrum','analyze_connectivity','visualize_eeg']
    args: dict[str,Any]
    depends_on: list[str] = Field(default_factory=list)

class Plan(StrictModel):
    steps: list[Step] = Field(min_length=1,max_length=12)

    @model_validator(mode='after')
    def validate_dependencies(self):
        names = [s.id for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError('Step IDs must be unique')
        seen = set()
        for s in self.steps:
            if s.id in s.depends_on or any(dep not in seen for dep in s.depends_on):
                raise ValueError(f'{s.id} depends on nonexistent or future step')
            seen.add(s.id)
        return self
