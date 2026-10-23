"""Shared configurable EEG channel layout for the original four dataset adapters.

Configure with EEG_TARGET_CHANNELS=26|64|native|Fp1,F3,...|@channels.json
or EEG_TARGET_COUNT=N. No channel is silently synthesized.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

STANDARD_26 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4',
    'T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3',
    'Pz','P4','P8','O1','Oz','O2',
]
STANDARD_64 = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz',
    'C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','M2','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3',
    'POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2',
]


def positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    val = float(raw) if raw is not None else float(default)
    if not np.isfinite(val) or val <= 0:
        raise ValueError(f'{name} must be a positive finite number: {val}')
    return val


def target_request() -> str | None:
    a = os.getenv('EEG_TARGET_CHANNELS')
    b = os.getenv('EEG_TARGET_COUNT')
    if a and b:
        raise ValueError('Specify EEG_TARGET_CHANNELS or EEG_TARGET_COUNT, not both')
    return a.strip() if a else b.strip() if b else None


def spatial_subset(names: list[str], n: int) -> list[str]:
    """Deterministic coverage of a montage; original order in final output."""
    if n < 1 or n > len(names):
        raise ValueError(f'Number of target channels must be 1..{len(names)}')
    if n == len(names):
        return list(names)
    import mne
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']
    valid = [ch for ch in names if ch in pos and np.all(np.isfinite(pos[ch]))]
    if len(valid) < n:
        # A minority of aliases lack standard_1020 coordinates: retain stable order.
        return list(names[:n])
    coords = np.stack([np.asarray(pos[ch]) for ch in valid])
    # Deterministic farthest-point coverage, initially use most superior channel.
    center = coords.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(coords - center, axis=1)))]
    dist = np.linalg.norm(coords - coords[chosen[0]], axis=1)
    for _ in range(1, n):
        dist[chosen] = -1
        ix = int(np.argmax(dist))
        chosen.append(ix)
        dist = np.minimum(np.maximum(dist, 0), np.linalg.norm(coords - coords[ix], axis=1))
    result = {valid[i] for i in chosen}
    return [ch for ch in names if ch in result]


def resolve_target_names(
    defaults: list[str],
    available: list[str],
    *,
    hydrocel: bool = False,
    hydro_source_map: dict[str, str] | None = None,
) -> list[str]:
    """Choose target names; available lists semantic and optionally native sensors.

    If N is specified: 1..26 use standard-26 coverage, 27..64 use standard-64
    coverage. Above 64 native E* channels are added only for HydroCel recordings.
    This never fabricates EEG channels to satisfy a requested count.
    """
    request = target_request()
    names: list[str]
    if request is None:
        names = list(defaults)
    elif request.lower() in {'native', 'all'}:
        names = ([name for name in available if (name == 'Cz' or (name.startswith('E') and name[1:].isdigit()))]
                 if hydrocel else list(available))
    elif request.startswith('@'):
        path = Path(request[1:])
        if path.suffix.lower() == '.json':
            names = json.loads(path.read_text(encoding='utf-8'))
        else:
            names = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip() and not line.strip().startswith('#')]
    elif ',' in request:
        names = [name.strip() for name in request.split(',') if name.strip()]
    elif request.isdecimal():
        n = int(request)
        if n <= 0:
            raise ValueError('Channel count must be >= 1')
        if n == 26:
            names = STANDARD_26.copy()
        elif n == 64:
            names = STANDARD_64.copy()
        elif n < 26:
            names = spatial_subset(STANDARD_26, n)
        elif n < 64:
            names = spatial_subset(STANDARD_64, n)
        else:
            names = STANDARD_64.copy()
            if hydrocel:
                used = set((hydro_source_map or {}).values())
                names += [name for name in available if name.startswith('E') and name[1:].isdigit()
                          and name not in used and name not in names][:n - 64]
    else:
        names = [request]
    if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x for x in names):
        raise ValueError('Target channel configuration must be a nonempty list of names')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate target channel names are not allowed')
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            'Requested EEG channels absent from this dataset: '
            f'{missing}. Available: {len(available)}. Use an explicit compatible list; '
            'adding channels requires valid sensor coordinates and a separate interpolation policy.'
        )
    if request and request.isdecimal() and len(names) != int(request):
        raise ValueError(f'Requested {request} channels but only {len(names)} are available')
    return names


def hydro_mapping(targets: list[str], original_mapping: dict[str, str]) -> dict[str, str]:
    """Named 10-10 positions get verified fixed HydroCel mapping; E# stay native."""
    result = {name: original_mapping.get(name, name) for name in targets}
    if len(set(result.values())) != len(targets):
        raise ValueError('Two requested output channels resolve to the same HydroCel electrode')
    return result


def env_path(name: str, default):
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def effective_window_seconds(seconds: float, sfreq: float) -> float:
    samples = round(seconds * sfreq)
    if samples < 1:
        raise ValueError('Window duration must contain at least one sample')
    return samples / sfreq
