"""Synthetic tests; no private EEG dataset files needed."""
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_universal import Settings, preprocess_raw, harmonize_raw, make_windows, pearson_fc, load_raw, load_blocks
from channel_layout import resolve_target_names, STANDARD_26, STANDARD_64, spatial_subset


def make_raw(channels=8, sfreq=250., seconds=12.):
    t = np.arange(round(sfreq*seconds)) / sfreq
    x = np.stack([5e-6 * np.sin(2*np.pi*(8+i)*t + i/3) for i in range(channels)])
    # Standard-1020 named EEG channels let montage and EOG tests run.
    names = STANDARD_26[:channels]
    info = mne.create_info(names, sfreq, 'eeg')
    return mne.io.RawArray(x, info, verbose='ERROR')


@pytest.mark.parametrize('channels,target,seconds',[(8,4,1.0),(8,8,2.0),(8,3,0.7),(1,1,1.3),(8,6,0.02)])
def test_shaping(channels,target,seconds):
    raw = make_raw(channels=channels, seconds=12.)
    cfg = Settings(channel_count=target, window_seconds=seconds, use_ica=False)
    arrays, info = preprocess_raw(raw, cfg)
    assert arrays['data'].shape[1:] == (target, round(seconds * 250))
    assert arrays['fc'].shape == (len(arrays['data']), target, target)
    assert len(arrays['channel_names']) == target
    assert np.allclose(np.diagonal(arrays['fc'], axis1=1, axis2=2), 0)
    assert np.allclose(arrays['window_end_sec']-arrays['window_start_sec'],
                       round(seconds*250)/250)
    assert np.all(np.isfinite(arrays['data']))
    assert np.all(np.isfinite(arrays['fc']))
    json.loads(str(arrays['metadata_json']))


def test_event_boundaries_and_stride():
    raw = make_raw(seconds=12.)
    cfg = Settings(channels=tuple(STANDARD_26[:4]), window_seconds=1.5,
                   stride_seconds=0.5, use_ica=False)
    a, _ = preprocess_raw(raw, cfg, event_blocks=[(0., 4., 'EO'), (6., 12., 'EC')])
    assert all(e <= 4 or s >= 6 for s,e in zip(a['window_start_sec'],a['window_end_sec']))
    assert set(a['labels']) == {'EO','EC'}
    assert 'block_000' not in a['event_block_ids'][a['labels']=='EC']


def test_missing_default_error():
    cfg = Settings(channels=('Fp1','Fp2','Cz'), use_ica=False)
    with pytest.raises(ValueError, match='Missing EEG channels'):
        harmonize_raw(make_raw(2), cfg)


def test_missing_with_explicit_interpolation():
    cfg = Settings(channels=('Fp1','Fp2','Cz'), interpolate_missing=True, use_ica=False)
    raw, details = harmonize_raw(make_raw(4), cfg)
    assert raw.ch_names == ['Fp1','Fp2','Cz']
    assert details['virtual_interpolated_channels'][0]['channel'] == 'Cz'


def test_numeric_npz_roundtrip(tmp_path):
    raw = make_raw(channels=4)
    p=tmp_path/'sample.npz'
    np.savez(p,data=raw.get_data()*1e6, sfreq=250.,channel_names=raw.ch_names)
    loaded=load_raw(p)
    np.testing.assert_allclose(raw.get_data(),loaded.get_data())


def test_invalid_settings_and_unsupported_expansion():
    with pytest.raises(ValueError):
        Settings(window_seconds=-1.)
    with pytest.raises(ValueError):
        Settings(channel_count=0)
    with pytest.raises(ValueError):
        harmonize_raw(make_raw(4), Settings(channel_count=5))


def test_harmonized_old_adapters(monkeypatch):
    # Configuration resolved dynamically at module import; no actual dataset reading.
    from importlib.util import spec_from_file_location, module_from_spec
    base=Path(__file__).resolve().parents[1]
    monkeypatch.setenv('EEG_TARGET_CHANNELS','26')
    monkeypatch.setenv('EEG_WINDOW_SECONDS','1.25')
    for filename, expected in [('PREDCT_FINAL.py','COMMON_64'),('HBN_FINAL.py','COMMON_64'),
                               ('MODMA_FINAL.py','COMMON_64'),('TDBRAIN_FINAL.py','EEG_26')]:
        spec=spec_from_file_location('_test_'+filename.replace('.','_'),base/filename)
        mod=module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert getattr(mod,expected)==STANDARD_26
        assert mod.WINDOW_SECONDS==1.25
        assert mod.TARGET_SFREQ==250.
        if filename in ['MODMA_FINAL.py','HBN_FINAL.py']:
            assert len(mod.COMMON64_TO_HYDROCEL)==26


def test_layout_count_presets():
    assert resolve_target_names(STANDARD_64,STANDARD_64)==STANDARD_64
    assert len(spatial_subset(STANDARD_64,32))==32
    assert len(set(spatial_subset(STANDARD_64,32)))==32


def test_original_modma_window_samples(monkeypatch):
    # Detect time-index drift when duration * sfreq is not an integer.
    import importlib.util
    monkeypatch.setenv('EEG_TARGET_CHANNELS','26')
    monkeypatch.setenv('EEG_WINDOW_SECONDS','0.733')
    for filename in ['MODMA_FINAL.py','TDBRAIN_FINAL.py']:
        spec=importlib.util.spec_from_file_location('_dynamic_'+filename.replace('.','_'),
                                                     Path(__file__).resolve().parents[1] / filename)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw=make_raw(channels=8)
        expected=list(module.COMMON_64 if filename.startswith('MODMA') else module.EEG_26)
        # Build a 26-channel signal for the native 26-channel path.
        sfreq=250.
        x=np.tile(raw.get_data()[0:1],(len(expected),1))
        raw=mne.io.RawArray(x,mne.create_info(expected, sfreq, 'eeg'),verbose='ERROR')
        data,starts,ends,*rest=module.make_one_second_ec_windows(raw)
        actual=round(module.WINDOW_SECONDS * sfreq)/sfreq
        assert data.shape[1:] == (26,round(module.WINDOW_SECONDS*sfreq))
        np.testing.assert_allclose(ends-starts,actual)
        np.testing.assert_allclose(starts,np.arange(len(starts))*actual)


def test_hydrocel_129_channel_selection(monkeypatch):
    import importlib.util
    base=Path(__file__).resolve().parents[1]
    for request in ('80','129','native'):
        monkeypatch.setenv('EEG_TARGET_CHANNELS',request)
        spec=importlib.util.spec_from_file_location('test_hbn_'+request,base/'HBN_FINAL.py')
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        n=129 if request=='native' else int(request)
        assert len(mod.COMMON_64)==n
        assert len(set(mod.COMMON64_TO_HYDROCEL.values()))==n


def test_zero_duration_annotation_rejected():
    raw=make_raw(4)
    raw.set_annotations(mne.Annotations([0.,2.],[0.,0.],['EO','EC']))
    with pytest.raises(ValueError, match='instantaneous markers'):
        load_blocks(None,raw,'Unknown')
