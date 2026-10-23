"""Dataset-independent EEG preprocessing (MNE Raw or numeric NxT input).

Exports one NPZ per recording, retaining event boundaries and channel metadata.
Only physically observed channels are used by default. No synthetic electrodes
are created unless missing-channel interpolation is explicitly enabled.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy.io import loadmat


@dataclass
class Settings:
    target_sfreq: float = 250.0
    window_seconds: float = 2.0
    stride_seconds: float | None = None
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    notch_freqs: tuple[float, ...] = (50.0, 60.0)
    filter_order: int = 4
    channels: tuple[str, ...] | None = None
    channel_count: int | None = None
    interpolate_missing: bool = False
    bad_z_threshold: float = 6.0
    interpolation_neighbors: int = 3
    use_ica: bool = True
    ica_max_components: int = 10
    ica_max_ocular: int = 2
    ica_fit_seconds: float = 30.0
    ica_random_state: int = 97
    max_abs_uv: float = 200.0
    max_p2p_uv: float = 400.0
    min_rms_uv: float = 0.05
    window_robust_z: float = 8.0
    baseline: bool = True
    default_label: str = 'Unknown'

    def __post_init__(self):
        for name in ('target_sfreq', 'window_seconds', 'bandpass_high', 'interpolation_neighbors'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.stride_seconds is not None and (not np.isfinite(self.stride_seconds) or self.stride_seconds <= 0):
            raise ValueError('stride_seconds must be positive and finite')
        if self.bandpass_low < 0 or self.bandpass_low >= self.bandpass_high:
            raise ValueError('Invalid filter band')
        if self.target_sfreq <= 2 * self.bandpass_high:
            raise ValueError('Target Nyquist frequency must exceed bandpass_high')
        if self.channel_count is not None and self.channel_count < 1:
            raise ValueError('channel_count must be positive')
        if self.channel_count is not None and self.channels is not None:
            raise ValueError('Specify channel_count or explicit channels, not both')
        if self.channels is not None and (not self.channels or len(self.channels) != len(set(self.channels))):
            raise ValueError('Explicit channels cannot be empty or duplicated')


def parse_names(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    if value.startswith('@'):
        path = Path(value[1:])
        if path.suffix.lower() == '.json':
            names = json.loads(path.read_text(encoding='utf8'))
        else:
            names = [line.strip() for line in path.read_text(encoding='utf8').splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
    else:
        names = [x.strip() for x in value.split(',') if x.strip()]
    if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
        raise ValueError('Channel file must be a JSON array or newline-separated text')
    return tuple(names)


def _maybe_attach_montage(raw: mne.io.BaseRaw, montage: str | None):
    if montage:
        raw.set_montage(montage, on_missing='ignore', verbose='ERROR')
    elif raw.get_montage() is None:
        names = raw.ch_names
        candidate = 'GSN-HydroCel-129' if any(x.startswith('E') and x[1:].isdigit() for x in names) else 'standard_1020'
        raw.set_montage(candidate, on_missing='ignore', verbose='ERROR')


def _positions(raw: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    pos = {}
    for idx in mne.pick_types(raw.info, eeg=True, exclude=[]):
        xyz = np.asarray(raw.info['chs'][idx]['loc'][:3], dtype=float)
        if np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e-8:
            pos[raw.ch_names[idx]] = xyz
    return pos


def _idw(coords: dict[str, np.ndarray], target: str, available: Sequence[str], k: int):
    if target not in coords:
        raise ValueError(f'No electrode position for {target}; cannot interpolate')
    scored = sorted(((float(np.linalg.norm(coords[c] - coords[target])), c)
                     for c in available if c in coords and c != target))
    if not scored:
        raise ValueError(f'No positioned source channels for interpolating {target}')
    scored = scored[:min(k, len(scored))]
    distances = np.array([d for d, _ in scored])
    if distances[0] < 1e-10:
        weights = np.array([1.0] + [0.0] * (len(scored) - 1))
    else:
        weights = 1 / np.maximum(distances, 1e-8)
        weights /= weights.sum()
    return [ch for _, ch in scored], weights


def _select_evenly(raw: mne.io.BaseRaw, count: int) -> list[str]:
    # Prefer spatial coverage; fall back to stable order for unlabeled arrays.
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if count > len(names):
        raise ValueError(f'Requested {count} channels but input contains only {len(names)} EEG sensors. '
                         'Specify explicit target names and --interpolate-missing to synthesize missing positions.')
    if count == len(names):
        return names
    pos = _positions(raw)
    if len(pos) < count or any(ch not in pos for ch in names):
        return names[:count]
    coords = np.stack([pos[name] for name in names])
    center = np.mean(coords, axis=0)
    selected = [int(np.argmax(np.linalg.norm(coords - center, axis=1)))]
    nearest = np.linalg.norm(coords - coords[selected[0]], axis=1)
    for _ in range(1, count):
        nearest[selected] = -1.0
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(np.maximum(nearest, 0), np.linalg.norm(coords - coords[index], axis=1))
    keep = set(selected)
    return [name for i, name in enumerate(names) if i in keep]


def harmonize_raw(raw: mne.io.BaseRaw, cfg: Settings, *, montage: str | None = None,
                  source_map: dict[str, str] | None = None):
    """Produce raw with EEG channels in chosen order, plus EOG until after ICA.

    source_map maps desired target names -> existing source names, e.g. Fp1:E22.
    """
    raw = raw.copy().load_data()
    # The four original datasets encode ocular sensors under several aliases.
    known_eog = {'HEOG', 'VEOG', 'VPVA', 'VNVB', 'HPHL', 'HNHR', 'EOG', 'EOG1', 'EOG2'}
    ocular = {name: 'eog' for name in raw.ch_names
              if name.upper() in known_eog or name.upper().startswith('EOG-')}
    if ocular:
        raw.set_channel_types(ocular, verbose='ERROR')
    src_count = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    if not src_count:
        raise ValueError('No EEG channels were identified in source')
    _maybe_attach_montage(raw, montage)
    if source_map:
        if len(set(source_map.values())) != len(source_map):
            raise ValueError('source_map must be one-to-one')
        missing_src = [src for src in source_map.values() if src not in raw.ch_names]
        if missing_src:
            raise ValueError(f'Mapped source channels missing: {missing_src}')
        raw.rename_channels({src: dest for dest, src in source_map.items()})
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if cfg.channels is not None:
        desired = list(cfg.channels)
    elif cfg.channel_count is not None:
        desired = _select_evenly(raw, cfg.channel_count)
    else:
        desired = names
    missing = [name for name in desired if name not in names]
    created = []
    if missing:
        if not cfg.interpolate_missing:
            raise ValueError(f'Missing EEG channels {missing}. Set --interpolate-missing only when '
                             'valid sensor coordinates are available; otherwise use an existing channel list.')
        montage_obj = raw.get_montage()
        source_pos = _positions(raw)
        if montage_obj is None:
            raise ValueError('Cannot interpolate absent targets without a montage')
        # standard montage can supply positions for requested, still preserving real-source positions
        all_pos = dict(source_pos)
        all_pos.update({k: np.asarray(v, float) for k, v in montage_obj.get_positions()['ch_pos'].items()
                        if np.all(np.isfinite(v))})
        standard = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
        for name in missing:
            if name in standard and name not in all_pos:
                all_pos[name] = np.asarray(standard[name])
        if not all(x in all_pos for x in missing):
            raise ValueError(f'Target channels have no spatial coordinates: {[x for x in missing if x not in all_pos]}')
        additional = []
        for name in missing:
            neighbors, weights = _idw(all_pos, name, names, cfg.interpolation_neighbors)
            additional.append(np.average(raw.get_data(picks=neighbors), axis=0, weights=weights))
            created.append({'channel': name, 'sources': neighbors,
                            'weights': [float(w) for w in weights]})
        # Raw.add_channels accepts a separately constructed RawArray.
        info = mne.create_info(missing, raw.info['sfreq'], ['eeg'] * len(missing))
        synthetic = mne.io.RawArray(np.stack(additional), info, verbose='ERROR')
        raw.add_channels([synthetic], force_update_info=True)
        raw.set_montage(mne.channels.make_dig_montage(
            ch_pos={n: all_pos[n] for n in desired if n in all_pos}, coord_frame='head'),
            on_missing='ignore', verbose='ERROR')
    aux = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True, exclude=[])]
    raw.pick(desired + aux)
    raw.reorder_channels(desired + aux)
    return raw, {'source_eeg_count': src_count, 'selected_eeg_count': len(desired),
                 'channel_names': desired, 'virtual_interpolated_channels': created}


def bad_channel_repair(raw: mne.io.BaseRaw, cfg: Settings):
    eeg = raw.copy().pick('eeg')
    x = eeg.get_data(stop=min(eeg.n_times, int(30 * eeg.info['sfreq'])))
    sd = x.std(axis=1)
    logs = np.log10(sd + np.finfo(float).eps)
    med = np.median(logs)
    mad = np.median(np.abs(logs - med))
    z = np.zeros_like(logs) if mad < np.finfo(float).eps else 0.67448975 * (logs - med) / mad
    bad = [eeg.ch_names[i] for i in np.flatnonzero((sd < 1e-12) | (np.abs(z) > cfg.bad_z_threshold))]
    positions = _positions(raw)
    repaired = {}
    unresolved = []
    all_eeg = eeg.ch_names
    for ch in bad:
        good = [name for name in all_eeg if name not in bad and name in positions]
        if ch not in positions or not good:
            unresolved.append(ch)
            continue
        neighbors, weights = _idw(positions, ch, good, cfg.interpolation_neighbors)
        raw._data[raw.ch_names.index(ch)] = np.average(raw.get_data(picks=neighbors), axis=0, weights=weights)
        repaired[ch] = neighbors
    if unresolved:
        warnings.warn(f'Bad EEG channels could not be interpolated: {unresolved}', RuntimeWarning)
    return {'bad_channels': bad, 'repaired_channels': repaired,
            'unresolved_channels': unresolved, 'robust_z': {name: float(val) for name, val in zip(all_eeg, z)}}


def remove_ocular(raw: mne.io.BaseRaw, cfg: Settings):
    eeg_count = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True, exclude=[])]
    if not cfg.use_ica or eeg_count < 3 or not eog:
        return raw, {'performed': False, 'removed': [], 'reason': 'disabled_or_insufficient_eeg_or_no_eog'}
    fit = raw.copy()
    if fit.n_times / fit.info['sfreq'] > cfg.ica_fit_seconds:
        fit.crop(tmax=cfg.ica_fit_seconds, include_tmax=False)
    ica = ICA(n_components=min(cfg.ica_max_components, eeg_count - 1), method='fastica',
              random_state=cfg.ica_random_state, max_iter=300, verbose='ERROR')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ica.fit(fit, picks='eeg', verbose='ERROR')
        if any('converg' in str(w.message).lower() for w in caught):
            return raw, {'performed': True, 'removed': [], 'reason': 'non_converged'}
        candidates = {}
        for ch in eog:
            try:
                inds, scores = ica.find_bads_eog(fit, ch_name=ch, threshold=2.5, verbose='ERROR')
                for i in inds:
                    score = abs(float(scores[i]))
                    if score >= 0.6:
                        candidates[i] = max(score, candidates.get(i, 0.0))
            except Exception as exc:
                warnings.warn(f'Unable to score EOG {ch}: {exc}', RuntimeWarning)
        remove = sorted(candidates, key=candidates.get, reverse=True)[:cfg.ica_max_ocular]
        if remove:
            ica.exclude = remove
            raw = ica.apply(raw.copy(), verbose='ERROR')
        return raw, {'performed': True, 'removed': [int(i) for i in remove], 'reason': 'ok'}
    except Exception as exc:
        warnings.warn(f'ICA skipped: {exc}', RuntimeWarning)
        return raw, {'performed': False, 'removed': [], 'reason': f'ica_error: {exc}'}


def load_blocks(path: Path | None, raw: mne.io.BaseRaw, default_label: str):
    """Return (start, end, label) blocks. TSV must provide onset,duration.

    If event TSV is supplied but invalid, fail closed; never silently treat as
    continuous and let windows bridge different physiological states.
    """
    if path:
        df = pd.read_csv(path, sep='\t')
        if not {'onset', 'duration'}.issubset(df.columns):
            raise ValueError('Events TSV needs onset and duration columns')
        field = next((key for key in ('trial_type', 'event_label', 'value') if key in df), None)
        blocks = [(float(row.onset), float(row.onset) + float(row.duration),
                   str(getattr(row, field)) if field else default_label)
                  for row in df.itertuples(index=False)]
    elif raw.annotations and len(raw.annotations):
        blocks = [(float(t), float(t + d), str(label)) for t, d, label in
                  zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description) if d > 0]
        if not blocks:
            raise ValueError('Annotations contain only instantaneous markers (duration=0). '
                             'Supply an onset/duration --events-tsv, use a dataset-specific adapter, '
                             'or deliberately set --ignore-annotations to process the entire recording.')
    else:
        blocks = [(0., raw.n_times / raw.info['sfreq'], default_label)]
    blocks.sort(key=lambda x: x[0])
    for i, (start, end, label) in enumerate(blocks):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(f'Invalid event interval: {blocks[i]}')
        if i and start < blocks[i-1][1] - 1e-8:
            raise ValueError(f'Overlapping event intervals {blocks[i-1]} / {blocks[i]}')
    return blocks


def make_windows(raw: mne.io.BaseRaw, cfg: Settings, blocks):
    sfreq = float(raw.info['sfreq'])
    ns = int(round(cfg.window_seconds * sfreq))
    stride = int(round((cfg.stride_seconds or cfg.window_seconds) * sfreq))
    if ns < 2 or stride < 1:
        raise ValueError('Window must contain at least two samples for Pearson FC; stride at least one sample')
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    x = raw.get_data(picks=names)
    windows = []
    starts = []
    ends = []
    labels = []
    blockids = []
    for blockid, (start, end, label) in enumerate(blocks):
        low = max(0, int(np.ceil(start * sfreq - 1e-9)))
        high = min(x.shape[1], int(np.floor(end * sfreq + 1e-9)))
        for i in range(low, high - ns + 1, stride):
            piece = x[:, i:i+ns].copy()
            if cfg.baseline:
                piece -= np.mean(piece, axis=1, keepdims=True)
            windows.append(np.asarray(piece * 1e6, dtype=np.float32))
            starts.append(i / sfreq)
            ends.append((i+ns) / sfreq)
            labels.append(label)
            blockids.append(blockid)
    if not windows:
        raise ValueError(f'No complete {cfg.window_seconds}s window in the available event intervals')
    return np.stack(windows), np.asarray(starts), np.asarray(ends), np.asarray(labels), np.asarray(blockids)


def window_qc(data: np.ndarray, cfg: Settings):
    maxabs = np.max(np.abs(data), axis=(1,2))
    p2p = np.max(np.ptp(data, axis=2), axis=1)
    rms = np.sqrt(np.mean(np.square(data.astype(np.float64)), axis=(1,2)))
    keep = ((maxabs <= cfg.max_abs_uv) & (p2p <= cfg.max_p2p_uv) & (rms >= cfg.min_rms_uv))
    if len(data) >= 20:
        logs = np.log(np.maximum(rms, 1e-8))
        med = np.median(logs)
        mad = np.median(abs(logs - med))
        if mad > 1e-12:
            keep &= abs(0.67448975 * (logs - med) / mad) <= cfg.window_robust_z
    return keep, {'total': len(data), 'kept': int(keep.sum()), 'rejected': int((~keep).sum())}


def pearson_fc(windows: np.ndarray) -> np.ndarray:
    # FC computation must derive channels dynamically from windows.shape[1].
    x = windows.astype(np.float32) - windows.mean(axis=2, keepdims=True)
    norms = np.sqrt((x*x).sum(axis=2, dtype=np.float64)).astype(np.float32)
    xn = np.divide(x, norms[:,:,None], out=np.zeros_like(x), where=norms[:,:,None] > 1e-20)
    fc = np.einsum('nct,ndt->ncd', xn, xn, optimize=True)
    fc = np.clip(fc, -1, 1)
    diag = np.arange(x.shape[1])
    fc[:,diag,diag] = 0
    return fc.astype(np.float32)


def preprocess_raw(raw: mne.io.BaseRaw, cfg: Settings, *, event_blocks=None,
                   montage: str | None = None, source_map: dict[str, str] | None = None):
    """Return (arrays, metadata) without side effects on caller's Raw."""
    raw, channels_info = harmonize_raw(raw, cfg, montage=montage, source_map=source_map)
    raw._data[~np.isfinite(raw._data)] = 0
    input_sfreq = float(raw.info['sfreq'])
    notch = [x for x in cfg.notch_freqs if x < input_sfreq/2 and x > cfg.bandpass_high]
    for line_hz in notch:
        raw.notch_filter([line_hz], method='iir', verbose='ERROR')
    if cfg.bandpass_high >= input_sfreq / 2:
        raise ValueError(f'Input sample rate {input_sfreq:g} Hz too low for {cfg.bandpass_high:g} Hz low-pass')
    raw.filter(cfg.bandpass_low, cfg.bandpass_high, method='iir',
               iir_params={'order': cfg.filter_order, 'ftype': 'butter'}, verbose='ERROR')
    if not np.isclose(raw.info['sfreq'], cfg.target_sfreq):
        raw.resample(cfg.target_sfreq, verbose='ERROR')
    bad = bad_channel_repair(raw, cfg)
    n_eeg = channels_info['selected_eeg_count']
    if n_eeg >= 2:
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')
    else:
        warnings.warn('Single-channel CAR is undefined; referencing skipped', RuntimeWarning)
    raw, ica_meta = remove_ocular(raw, cfg)
    if event_blocks is None:
        event_blocks = load_blocks(None, raw, cfg.default_label)
    # Annotation onset / duration is in seconds and remains valid after resampling.
    data, starts, ends, labels, ids = make_windows(raw, cfg, event_blocks)
    keep, qc = window_qc(data, cfg)
    if not keep.any():
        raise ValueError('All EEG windows rejected by artifact QC; inspect input units, channels, and thresholds')
    data, starts, ends, labels, ids = (arr[keep] for arr in (data, starts, ends, labels, ids))
    block_begin = np.asarray([event_blocks[int(i)][0] for i in ids], dtype=np.float64)
    block_end = np.asarray([event_blocks[int(i)][1] for i in ids], dtype=np.float64)
    out = {'data': data, 'fc': pearson_fc(data), 'labels': labels,
           'event_codes': np.zeros(len(ids), dtype=np.int16),
           'event_block_ids': np.asarray([f'block_{int(i):03d}' for i in ids]),
           'event_block_start_sec': block_begin, 'event_block_end_sec': block_end,
           'window_start_sec': starts, 'window_end_sec': ends,
           'window_times_sec': np.stack([starts, ends], axis=1),
           'channel_names': np.asarray(channels_info['channel_names']),
           'sfreq': np.asarray(cfg.target_sfreq, dtype=np.float32),
           'data_unit': np.asarray('uV'),
           'channel_positions_xyz': np.asarray([raw.info['chs'][raw.ch_names.index(n)]['loc'][:3]
               for n in channels_info['channel_names']], dtype=np.float64)}
    meta = {'input_sfreq': input_sfreq, 'output_sfreq': cfg.target_sfreq,
            'requested_window_sec': cfg.window_seconds,
            'effective_window_sec': data.shape[2] / cfg.target_sfreq,
            'effective_stride_sec': round((cfg.stride_seconds or cfg.window_seconds) * cfg.target_sfreq) / cfg.target_sfreq,
            'shape': list(data.shape), 'fc_shape': list(out['fc'].shape),
            'channel_info': channels_info, 'bad_channel_info': bad,
            'ica': ica_meta, 'window_qc': qc,
            'event_blocks': event_blocks, 'settings': asdict(cfg),
            'notch_applied_hz': notch, 'reference': 'CAR' if n_eeg >= 2 else 'unchanged (single channel)'}
    out['metadata_json'] = np.asarray(json.dumps(meta, ensure_ascii=False, default=str))
    return out, meta


def load_raw(path: Path, *, input_sfreq: float | None = None,
             input_channels: Sequence[str] | None = None, mat_unit: str = 'uV'):
    """Supported input: FIF/EDF/BDF/SET/VHDR/CNT, NPY/NPZ NxT, MAT EEG struct or 2D data."""
    path = Path(path)
    suffix = path.suffix.lower()
    readers = {'.fif': mne.io.read_raw_fif, '.edf': mne.io.read_raw_edf,
               '.bdf': mne.io.read_raw_bdf, '.set': mne.io.read_raw_eeglab,
               '.vhdr': mne.io.read_raw_brainvision, '.cnt': mne.io.read_raw_cnt}
    if suffix in readers:
        return readers[suffix](path, preload=True, verbose='ERROR')
    if suffix == '.npy':
        data = np.load(path, allow_pickle=False)
        sfreq = input_sfreq
        names = input_channels
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            data = z['data']
            sfreq = input_sfreq if input_sfreq is not None else float(z['sfreq']) if 'sfreq' in z else None
            names = input_channels if input_channels is not None else list(z['channel_names'].astype(str)) if 'channel_names' in z else None
    elif suffix == '.mat':
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if 'EEG' in mat:
            eeg = mat['EEG']
            data = np.asarray(eeg.data)
            sfreq = input_sfreq if input_sfreq is not None else float(np.asarray(eeg.srate).squeeze())
            if input_channels is not None:
                names = input_channels
            else:
                locs = getattr(eeg, 'chanlocs', None)
                if locs is None:
                    names = None
                else:
                    flat = np.asarray(locs).reshape(-1)
                    names = [str(getattr(item, 'labels', f'EEG{i+1}')) for i, item in enumerate(flat)]
        elif 'data' in mat:
            data = np.asarray(mat['data'])
            sfreq = input_sfreq if input_sfreq is not None else float(np.asarray(mat.get('sfreq', mat.get('srate', np.nan))).squeeze())
            names = input_channels
        else:
            raise ValueError('MAT should contain EEG.data/srate or data+sfreq')
    else:
        raise ValueError(f'Unsupported file type: {suffix}')
    if data.ndim != 2:
        raise ValueError(f'Expected continuous (channels,time), got {data.shape}; not pre-windowed data')
    if names is not None and len(names) == data.shape[1] and len(names) != data.shape[0]:
        data = data.T
    if names is None:
        # For unlabeled NPY, assume channels-first. Provide --input-channels to disambiguate.
        names = [f'EEG{i+1}' for i in range(data.shape[0])]
    if len(names) != data.shape[0]:
        raise ValueError(f'Channel names length {len(names)} does not match data {data.shape}')
    if sfreq is None or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError('Numeric input requires valid --input-sfreq or sfreq in file')
    factor = {'v': 1.0, 'mv': 1e-3, 'uv': 1e-6}[mat_unit.lower()]
    info = mne.create_info(list(names), sfreq, ['eeg'] * len(names))
    return mne.io.RawArray(data.astype(np.float64) * factor, info, verbose='ERROR')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Universal EEG preprocessing, original pipeline-inspired')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=Path)
    group.add_argument('--input-dir', type=Path)
    parser.add_argument('--glob', default='*.edf', help='Input-dir pattern; e.g. **/*.set')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-channels', help='Comma list or @channels.json / @channels.txt')
    parser.add_argument('--target-count', type=int)
    parser.add_argument('--window-seconds', type=float, default=2.0)
    parser.add_argument('--stride-seconds', type=float, default=None)
    parser.add_argument('--sfreq', type=float, default=250.0)
    parser.add_argument('--bandpass-low', type=float, default=0.5)
    parser.add_argument('--bandpass-high', type=float, default=45.0)
    parser.add_argument('--notch', type=float, nargs='*', default=[50., 60.])
    parser.add_argument('--montage', help='MNE standard montage name')
    parser.add_argument('--source-map', type=Path, help='JSON {"target":"source"}')
    parser.add_argument('--interpolate-missing', action='store_true')
    parser.add_argument('--events-tsv', type=Path, help='BIDS onset/duration/trial_type; single input only')
    parser.add_argument('--default-label', default='Unknown')
    parser.add_argument('--dataset-name', default='GENERIC', help='Dataset tag in output NPZ')
    parser.add_argument('--subject-id', help='Explicit subject ID (single file); otherwise BIDS sub-* / stem')
    parser.add_argument('--ignore-annotations', action='store_true',
                        help='Explicitly treat complete file as one block; ignores event markers')
    parser.add_argument('--eog-channels', help='Comma-separated EOG channel names')
    parser.add_argument('--input-sfreq', type=float, help='Sampling frequency for NPY/MAT if missing')
    parser.add_argument('--input-channels', help='Comma names / @file for NPY/MAT')
    parser.add_argument('--mat-unit', choices=['V','mV','uV'], default='uV')
    parser.add_argument('--disable-ica', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--visualize', action='store_true', help='Generate figures and HTML index from output NPZ')
    parser.add_argument('--viz-plots', default='default', help='default / all / comma-separated visualization names')
    parser.add_argument('--viz-max-windows', type=int, default=64)
    parser.add_argument('--viz-dpi', type=int, default=180)
    args = parser.parse_args(argv)
    cfg = Settings(target_sfreq=args.sfreq, window_seconds=args.window_seconds,
                   stride_seconds=args.stride_seconds, bandpass_low=args.bandpass_low,
                   bandpass_high=args.bandpass_high, notch_freqs=tuple(args.notch),
                   channels=parse_names(args.target_channels), channel_count=args.target_count,
                   interpolate_missing=args.interpolate_missing, default_label=args.default_label,
                   use_ica=not args.disable_ica)
    sources = [args.input_file] if args.input_file else sorted(args.input_dir.glob(args.glob))
    if not sources:
        parser.error('No input files found')
    if args.ignore_annotations and args.events_tsv:
        parser.error('Cannot combine --ignore-annotations with --events-tsv')
    if args.subject_id and len(sources) != 1:
        parser.error('--subject-id applies to one --input-file only')
    if args.events_tsv and len(sources) != 1:
        parser.error('Use --events-tsv only with one --input-file')
    source_map = json.loads(args.source_map.read_text(encoding='utf8')) if args.source_map else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        # Preserve source hierarchy in batch mode to avoid clobbering same-stem
        # recordings from different subjects, e.g. sub-01/rest.edf, sub-02/rest.edf.
        relative_parent = source.relative_to(args.input_dir).parent if args.input_dir else Path('.')
        dest = (args.output_dir / relative_parent / (source.stem + '_preprocessed.npz')).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not args.overwrite:
            print(f'SKIP {dest} (use --overwrite)')
            continue
        try:
            raw = load_raw(source, input_sfreq=args.input_sfreq,
                           input_channels=parse_names(args.input_channels), mat_unit=args.mat_unit)
            if args.eog_channels:
                eog = parse_names(args.eog_channels)
                raw.set_channel_types({ch:'eog' for ch in eog if ch in raw.ch_names}, verbose='ERROR')
            blocks = ([(0., raw.n_times / raw.info['sfreq'], cfg.default_label)]
                      if args.ignore_annotations else load_blocks(args.events_tsv, raw, cfg.default_label))
            arrays, meta = preprocess_raw(raw, cfg, event_blocks=blocks,
                                          montage=args.montage, source_map=source_map)
            import re
            m = re.search(r'sub-[A-Za-z0-9]+', str(source))
            subject_id = args.subject_id or (m.group() if m else source.stem)
            arrays['subject_ids'] = np.asarray([subject_id]*len(arrays['data']))
            arrays['run_ids'] = np.asarray([source.stem]*len(arrays['data']))
            arrays['dataset_names'] = np.asarray([args.dataset_name]*len(arrays['data']))
            arrays['source_file'] = np.asarray(str(source))
            meta.update({'subject_id': subject_id, 'dataset_name': args.dataset_name,
                         'source_file': str(source)})
            arrays['metadata_json'] = np.asarray(json.dumps(meta, ensure_ascii=False, default=str))
            np.savez_compressed(dest, **arrays)
            results.append({'source': str(source), 'file': str(dest), 'status':'OK', 'shape':meta['shape']})
            print(f'OK {source.name}: {meta["shape"]} => {dest}')
        except Exception as exc:
            results.append({'source': str(source), 'status':'ERROR', 'error':str(exc)})
            print(f'ERROR {source}: {exc}')
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),encoding='utf8')
    if args.visualize and any(x['status']=='OK' for x in results):
        from eeg_visualization import main as visualization_main
        vis_args=['--manifest',str(manifest_path),'--output-dir',str(args.output_dir/'visualizations'),
                  '--plots',args.viz_plots,'--max-windows',str(args.viz_max_windows),
                  '--dpi',str(args.viz_dpi),'--compare']
        visualization_main(vis_args)
    if any(x['status'] == 'ERROR' for x in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
