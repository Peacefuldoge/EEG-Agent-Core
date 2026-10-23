"""Deterministic, framework-independent handcrafted EEG features.

All input amplitudes are converted to microvolts (uV) before computation.
Outputs have the layout (windows, channels, features). The Gaussian
*differential entropy* is 0.5*ln(2*pi*e*variance), expressed in nats when
voltage is numerically measured in uV. It is NOT discrete Shannon entropy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch

# Non-overlapping bands; a boundary frequency has zero measure in integration.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0),
}
UNIT_TO_UV = {'uv': 1.0, 'µv': 1.0, 'μv': 1.0, 'mv': 1e3, 'v': 1e6}


@dataclass
class FeatureMatrix:
    values: np.ndarray     # (windows, channels, features), in listed order
    feature_names: list[str]
    feature_units: list[str]
    sfreq: float
    channels: list[str]
    bands: dict[str, tuple[float, float]]
    total_band: tuple[float, float]
    band_de_method: str


def _checked_bands(bands: Mapping[str, tuple[float, float]], sfreq: float,
                   total_band: tuple[float, float]):
    if not bands or len(bands) > 32:
        raise ValueError('Supply between 1 and 32 frequency bands')
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError('sfreq must be positive and finite')
    nyquist = sfreq / 2.0
    lo, hi = map(float, total_band)
    if not np.isfinite([lo, hi]).all() or lo <= 0 or hi <= lo or hi >= nyquist:
        raise ValueError(f'total_band must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz)')
    checked: dict[str, tuple[float, float]] = {}
    for name, edges in bands.items():
        if not isinstance(name, str) or not name.isidentifier() or name.lower() != name:
            raise ValueError(f'Band name must be a lowercase Python identifier: {name!r}')
        if len(edges) != 2:
            raise ValueError('Each band requires [low, high]')
        a, b = map(float, edges)
        if not np.isfinite([a, b]).all() or a < lo or b > hi or b <= a:
            raise ValueError(f'Band {name}: require total_low <= low < high <= total_high')
        checked[name] = (a, b)
    # Overlap can be intentional (e.g. individualized bands), so not forbidden.
    return checked, (lo, hi)


def _integrated_band(freqs: np.ndarray, psd: np.ndarray, low: float, high: float):
    """Integrate Welch density (uV^2/Hz), interpolating band edges.

    Does not silently drop fractional FFT bins at arbitrary window sizes.
    """
    if not (0 <= low < high <= freqs[-1]):
        raise ValueError('Requested integration range outside the PSD frequencies')
    inner = np.flatnonzero((freqs > low) & (freqs < high))

    def at(f):
        idx = int(np.searchsorted(freqs, f, side='right'))
        if idx <= 0:
            return psd[..., 0]
        if idx >= len(freqs):
            return psd[..., -1]
        f0, f1 = freqs[idx-1], freqs[idx]
        w = (f - f0) / (f1 - f0)
        return psd[..., idx-1] * (1-w) + psd[..., idx] * w

    xs = np.r_[low, freqs[inner], high]
    ys = np.concatenate([at(low)[..., None], psd[..., inner], at(high)[..., None]], axis=-1)
    return np.maximum(trapezoid(ys, x=xs, axis=-1), 0.0)


def _gaussian_de(variance):
    """Gaussian differential entropy in nats; zero variance => NaN (degenerate)."""
    variance = np.asarray(variance, dtype=np.float64)
    return np.where(variance > 0, .5 * np.log(2*np.pi*np.e*np.maximum(variance, np.finfo(float).tiny)), np.nan)


def extract_handcrafted_features(
    data: np.ndarray,
    sfreq: float,
    *,
    channels: list[str] | None = None,
    data_unit: str = 'uV',
    bands: Mapping[str, tuple[float, float]] | None = None,
    total_band: tuple[float, float] = (1.0, 45.0),
    band_de_method: str = 'filtered',
    welch_seconds: float = 2.0,
    filter_order: int = 4,
) -> FeatureMatrix:
    """Extract per-window per-channel EEG features without LLM / UI / MNE.

    `data`: (windows, channels, samples), or (channels, samples) for one window.
    `data_unit`: 'V', 'mV', or 'uV' (both micro symbols accepted).
    Relative powers are fractions in [0,1] of Welch power in `total_band`.
    `filtered`: band DE from zero-phase Butterworth bandpassed window variance.
       Short windows insufficient for sosfiltfilt are rejected, not silently padded.
    `psd`: Gaussian DE using integrated Welch bandpower as a variance proxy;
       this is *not* a time-domain bandpassed-variance DE estimate.
    Log-power is natural log(uV²); zero power => NaN.
    """
    x = np.asarray(data)
    if x.ndim == 2:
        x = x[None, :, :]
    if x.ndim != 3 or min(x.shape) < 1 or x.shape[-1] < 8:
        raise ValueError('data must have shape (windows,channels,samples) with >=8 samples')
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError('EEG data must be numeric')
    if data_unit.strip().lower() not in UNIT_TO_UV:
        raise ValueError(f'Unsupported EEG unit: {data_unit!r}; expected V, mV, uV')
    x = np.asarray(x, dtype=np.float64) * UNIT_TO_UV[data_unit.strip().lower()]
    if not np.isfinite(x).all():
        raise ValueError('Non-finite EEG samples are not accepted')
    if band_de_method not in ('filtered', 'psd'):
        raise ValueError("band_de_method must be 'filtered' or 'psd'")
    if not np.isfinite(welch_seconds) or welch_seconds <= 0:
        raise ValueError('welch_seconds must be positive')
    if filter_order < 1 or filter_order > 12:
        raise ValueError('filter_order must be in [1,12]')
    bands2, total = _checked_bands(DEFAULT_BANDS if bands is None else bands, sfreq, total_band)
    if channels is None:
        channels = [f'Ch{i+1}' for i in range(x.shape[1])]
    if len(channels) != x.shape[1] or len(set(channels)) != len(channels):
        raise ValueError('Channel labels must match channel dimension without duplicates')

    nperseg = min(x.shape[-1], max(8, int(round(welch_seconds * sfreq))))
    freqs, psd = welch(x, fs=sfreq, axis=-1, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling='density', detrend='constant')
    psd = np.maximum(psd, 0.)
    total_power = _integrated_band(freqs, psd, *total)
    variance = np.var(x, axis=-1, ddof=0)
    x_centered = x - np.mean(x, axis=-1, keepdims=True)

    # All features are arrays of shape (windows,channels).
    values: list[np.ndarray] = []
    names: list[str] = []
    units: list[str] = []

    def add(name: str, val: np.ndarray, unit: str):
        names.append(name)
        values.append(np.asarray(val, dtype=float))
        units.append(unit)

    add('mean', x.mean(axis=-1), 'uV')
    add('std', np.sqrt(variance), 'uV')
    add('variance', variance, 'uV^2')
    add('rms', np.sqrt(np.mean(x*x, axis=-1)), 'uV')
    add('peak_to_peak', np.ptp(x, axis=-1), 'uV')
    add('line_length_mean', np.abs(np.diff(x, axis=-1)).mean(axis=-1), 'uV/sample')
    add('zero_crossing_rate', (x_centered[..., :-1] * x_centered[..., 1:] < 0).mean(axis=-1), 'fraction')
    add('de_gaussian', _gaussian_de(variance), 'nats (uV reference)')

    # Hjorth activity = sample variance, mobility = sqrt(var(dx)/var(x)),
    # complexity = mobility(dx)/mobility(x), with numerical derivative in uV/s.
    dx = np.diff(x, axis=-1) * sfreq
    var_dx = np.var(dx, axis=-1)
    d2x = np.diff(dx, axis=-1) * sfreq
    var_d2 = np.var(d2x, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        mobility = np.sqrt(np.divide(var_dx, variance, out=np.full_like(variance, np.nan), where=variance>0))
        mobility_dx = np.sqrt(np.divide(var_d2, var_dx, out=np.full_like(var_dx, np.nan), where=var_dx>0))
        complexity = np.divide(mobility_dx, mobility, out=np.full_like(mobility, np.nan), where=mobility>0)
    add('hjorth_activity', variance, 'uV^2')
    add('hjorth_mobility', mobility, '1/s')
    add('hjorth_complexity', complexity, 'dimensionless')

    spec_mask = (freqs >= total[0]) & (freqs <= total[1])
    spec = psd[..., spec_mask]
    spec_f = freqs[spec_mask]
    if not len(spec_f):
        raise ValueError('Welch spectrum has no samples inside total_band')
    add('peak_frequency', spec_f[spec.argmax(axis=-1)], 'Hz')
    spec_sum = np.sum(spec, axis=-1, keepdims=True)
    pnorm = np.divide(spec, spec_sum, out=np.zeros_like(spec), where=spec_sum>0)
    ent = -(np.where(pnorm>0, pnorm*np.log(np.maximum(pnorm,np.finfo(float).tiny)), 0)).sum(axis=-1)
    ent = ent / np.log(spec.shape[-1]) if spec.shape[-1]>1 else np.full_like(ent, np.nan)
    ent = np.where(spec_sum[...,0]>0, ent, np.nan)
    add('spectral_entropy', ent, 'normalized [0,1]')
    add('total_power', total_power, 'uV^2')

    for band, (low, high) in bands2.items():
        bp = _integrated_band(freqs, psd, low, high)
        relative = np.divide(bp, total_power, out=np.full_like(bp,np.nan), where=total_power>0)
        add(f'power_{band}', bp, 'uV^2')
        add(f'relative_power_{band}', relative, 'fraction')
        add(f'log_power_{band}', np.where(bp>0, np.log(np.maximum(bp,np.finfo(float).tiny)), np.nan), 'ln(uV^2)')
        if band_de_method == 'filtered':
            sos = butter(filter_order, [low, high], btype='bandpass', fs=sfreq, output='sos')
            # Check explicitly rather than pad or silently degrade for short windows.
            ntaps = 2 * len(sos) + 1 - min((sos[:,2]==0).sum(), (sos[:,5]==0).sum())
            padlen = 3 * ntaps
            if x.shape[-1] <= padlen:
                raise ValueError(f'Band DE filtering ({band}) requires >{padlen} samples; '
                                 'use longer windows or band_de_method="psd"')
            filtered = sosfiltfilt(sos, x, axis=-1)
            band_var = np.var(filtered, axis=-1)
        else:
            band_var = bp
        add(f'de_{band}', _gaussian_de(band_var), 'nats (uV reference)')

    return FeatureMatrix(np.stack(values, axis=-1), names, units, float(sfreq),
                         list(channels), bands2, total, band_de_method)


def read_feature_npz(path: str | Path, *, max_windows: int=32,
                     assume_unit: str='uV', return_metadata: bool=False):
    """Load legacy/new EEG-Agent NPZ without matplotlib or unsafe pickle.

    Select evenly spaced windows like the visualizer. Assumes uV for legacy files
    without data_unit; callers may explicitly override `assume_unit`.
    """
    with np.load(path, allow_pickle=False) as z:
        if 'data' not in z or 'sfreq' not in z:
            raise ValueError('Preprocessed NPZ requires data and sfreq')
        x = z['data']
        if x.ndim != 3 or min(x.shape) < 1:
            raise ValueError('NPZ data must be (windows, channels, samples)')
        sfreq = float(np.asarray(z['sfreq']).item())
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError('Invalid NPZ sampling frequency')
        if 'channel_names' in z:
            names = list(np.asarray(z['channel_names']).astype(str))
        else:
            names = [f'Ch{i+1}' for i in range(x.shape[1])]
        if len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError('Invalid NPZ channel names')
        unit = str(np.asarray(z['data_unit']).item()) if 'data_unit' in z else assume_unit
        if max_windows <= 0:
            raise ValueError('max_windows must be positive')
        idx = np.unique(np.rint(np.linspace(0, x.shape[0]-1,
                                          min(max_windows,x.shape[0]))).astype(int))
        selected = np.asarray(x[idx], dtype=float)
        metadata: dict[str, np.ndarray] = {}
        for key in ('labels', 'subject_ids', 'dataset_names', 'run_ids'):
            if key in z:
                vals = np.asarray(z[key])
                if vals.ndim != 1 or len(vals) != x.shape[0]:
                    raise ValueError(f'Invalid {key}: expected one value per EEG window')
                metadata[key] = vals[idx].astype(str)
        if 'window_times_sec' in z:
            times = np.asarray(z['window_times_sec'],dtype=float)
            if times.shape != (x.shape[0],2) or not np.all(np.isfinite(times)):
                raise ValueError('Invalid window_times_sec')
            metadata['window_times_sec'] = times[idx]
        elif 'window_start_sec' in z and 'window_end_sec' in z:
            start = np.asarray(z['window_start_sec'],dtype=float)
            end = np.asarray(z['window_end_sec'],dtype=float)
            if start.shape != (x.shape[0],) or end.shape != (x.shape[0],):
                raise ValueError('Invalid window_start/end_sec')
            metadata['window_times_sec'] = np.column_stack((start[idx],end[idx]))
    result = (selected, sfreq, names, idx, unit)
    return (*result, metadata) if return_metadata else result
