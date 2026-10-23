"""Scientific guardrails. Reject physically impossible settings; report study warnings.

Warnings are not silently corrected: the researcher must approve revised settings.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ScientificReview:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    def to_dict(self):
        return asdict(self) | {'valid': not self.errors}


def review_preprocessing(params: dict, *, input_sfreq: float | None = None) -> ScientificReview:
    errors, warnings = [], []
    fs = float(params.get('target_sfreq', 250))
    low, high = float(params.get('bandpass_low', .5)), float(params.get('bandpass_high', 45))
    window = float(params.get('window_seconds', 1))
    stride = float(params.get('stride_seconds') or window)
    if not (0 <= low < high < fs / 2):
        errors.append('Bandpass frequencies must satisfy 0 <= low < high < target Nyquist.')
    if stride <= 0 or window <= 0:
        errors.append('Window length and stride must be positive.')
    if stride > window:
        warnings.append('Stride exceeds window length; EEG intervals will be skipped.')
    if window * fs < 16:
        errors.append('Fewer than 16 target samples per window.')
    if window < 2 and low <= 1:
        warnings.append('Sub-2-second windows cannot stably resolve delta / very-low-frequency oscillations.')
    if low and window * low < 3:
        warnings.append('Window contains fewer than 3 cycles at the high-pass cutoff; edge effects can dominate.')
    if input_sfreq and fs > input_sfreq:
        warnings.append('Upsampling does not recover information above the original Nyquist frequency.')
    if bool(params.get('use_ica', True)) and window < .5:
        warnings.append('ICA needs adequate continuous training data, not only short epochs.')
    if params.get('interpolate_missing'):
        warnings.append('Interpolated channels are synthetic estimates and must be documented.')
    if any(not (0 < float(f) < fs/2) for f in params.get('notch_freqs', [50.,60.])):
        errors.append('Notch frequencies must be within the target Nyquist range.')
    return ScientificReview(tuple(errors),tuple(dict.fromkeys(warnings)))


def review_connectivity(method: str, *, n_windows: int, n_times: int, sfreq: float,
                        bands: dict[str,tuple[float,float]]) -> ScientificReview:
    errors, warnings = [], []
    nyq = sfreq/2
    for name,(lo,hi) in bands.items():
        if not (0 < lo < hi < nyq):
            errors.append(f'Band {name} must be within (0, {nyq:g}) Hz.')
        if n_times/sfreq < 2/lo:
            warnings.append(f'{name}: short windows can produce unstable low-frequency connectivity.')
    if method != 'pearson' and n_windows < 8:
        warnings.append('Frequency-domain connectivity from fewer than 8 epochs can be unreliable.')
    if n_times < 16:
        errors.append('Too few samples for reliable spectral estimates.')
    return ScientificReview(tuple(errors), tuple(dict.fromkeys(warnings)))
