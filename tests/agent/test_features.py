"""Numerical scientific regression tests: deterministic synthetic EEG, no LLM."""
from pathlib import Path
import json

import numpy as np
import pytest
from pydantic import ValidationError

from eeg_agent import DEFAULT_BANDS, extract_handcrafted_features
from eeg_agent.features import read_feature_npz
from eeg_agent.schema import Plan
from eeg_agent.tools import EEGTools


def test_gaussian_de_closed_form_and_units():
    rng = np.random.default_rng(2)
    samples = rng.normal(0, 5, size=(2, 3, 1024))
    a = extract_handcrafted_features(samples, 250)
    b = extract_handcrafted_features(samples / 1e6, 250, data_unit='V')
    np.testing.assert_allclose(a.values, b.values, rtol=1e-9, atol=1e-8)
    idx = a.feature_names.index('de_gaussian')
    variance = np.var(samples, axis=-1)
    np.testing.assert_allclose(a.values[...,idx], .5*np.log(2*np.pi*np.e*variance))
    assert a.values.shape == (2, 3, 14 + 4*len(DEFAULT_BANDS))
    assert a.feature_units[idx] == 'nats (uV reference)'


def test_alpha_sine_relative_power_and_band_de():
    fs=250
    t=np.arange(5*fs)/fs
    rng=np.random.default_rng(5)
    data=(20*np.sin(2*np.pi*10*t) + rng.normal(0,.05,size=t.shape))[None,None,:]
    res=extract_handcrafted_features(data,fs)
    k=lambda name: res.values[0,0,res.feature_names.index(name)]
    assert k('relative_power_alpha') > .95
    assert k('power_alpha') > k('power_beta')*100
    assert k('de_alpha') > k('de_beta')
    assert np.isclose(k('peak_frequency'), 10, atol=.5)
    assert 0 <= k('spectral_entropy') <= 1


def test_short_window_and_custom_bands():
    fs=250
    x=np.arange(20,dtype=float)[None,None,:]
    with pytest.raises(ValueError,match='longer windows'):
        extract_handcrafted_features(x,fs)
    a=extract_handcrafted_features(x,fs,band_de_method='psd',bands={'custom':(8,15)})
    assert 'de_custom' in a.feature_names
    assert a.values.shape == (1,1,18)
    with pytest.raises(ValueError,match='Nyquist'):
        extract_handcrafted_features(x,fs,total_band=(1,150))
    with pytest.raises(ValueError,match='Non-finite'):
        extract_handcrafted_features(np.array([[[float('nan')]*30]]),fs)


def test_zero_channels_are_marked_missing():
    res=extract_handcrafted_features(np.zeros((1,1,250)),250)
    k=lambda name: res.values[0,0,res.feature_names.index(name)]
    assert np.isnan(k('de_gaussian'))
    assert np.isnan(k('relative_power_alpha'))
    assert np.isnan(k('de_alpha'))
    assert k('power_alpha') == 0


def test_npz_unit_legacy_and_window_subset(tmp_path):
    path=tmp_path/'legacy.npz'
    np.savez(path,data=np.ones((9,2,250))*1e-5,sfreq=250,channel_names=['C3','C4'],data_unit='V')
    x,fs,names,idx,unit=read_feature_npz(path,max_windows=3)
    assert np.array_equal(idx,[0,4,8])
    assert x.shape==(3,2,250) and unit=='V' and fs==250 and names==['C3','C4']


def test_tool_registry_exports_and_relative_power(tmp_path):
    d=tmp_path/'raw';d.mkdir()
    t=np.arange(1000)/250
    np.savez(d/'subject.npz',data=np.stack([
        np.array([9*np.sin(2*np.pi*10*t),12*np.sin(2*np.pi*18*t)]) for _ in range(4)]),
        sfreq=250,channel_names=['F3','F4'],data_unit='uV',
        labels=['EO','EO','EC','EC'],subject_ids=['sub-1']*4,
        dataset_names=['synthetic']*4,window_times_sec=np.array([[i*4.,(i+1)*4.] for i in range(4)]))
    tools=EEGTools(d,tmp_path/'out')
    assert 'extract_features' in tools.registry
    spec=next(s for s in tools.openai_schemas() if s['function']['name']=='extract_features')
    assert 'source' in spec['function']['parameters']['properties']
    result=tools.invoke('extract_features',{'source':str(d/'subject.npz'),'max_windows':3})
    assert result['shape']==[3,2,34]
    assert Path(result['output_csv']).exists()
    assert Path(result['features_npz']).exists()
    assert Path(result['manifest_json']).exists()
    assert 'output_npz' not in result # protects LangGraph latest EEG reference
    with np.load(result['features_npz'],allow_pickle=False) as z:
        assert z['features'].shape == (3,2,34)
        assert list(z['channel_names'])==['F3','F4']
        assert list(z['labels'])==['EO','EC','EC']
        assert list(z['subject_ids'])==['sub-1']*3
        assert z['window_times_sec'].shape==(3,2)
    manifest=json.loads(Path(result['manifest_json']).read_text())
    assert manifest['amplitude_unit']=='uV'
    assert 'subject_ids' in manifest['preserved_window_metadata_fields']
    assert manifest['relative_power_denominator_hz']==[1.0,45.0]
    assert manifest['feature_units']['relative_power_alpha']=='fraction'
    assert result['summary_by_feature']['relative_power_alpha']['mean'] > .45
    assert result['summary_by_feature']['relative_power_beta']['mean'] > .45
    assert len(Path(result['output_csv']).read_text().splitlines()) == 7


def test_tool_path_isolation_and_plan_registration(tmp_path):
    d=tmp_path/'raw';d.mkdir()
    external=tmp_path/'private.npz'
    np.savez(external,data=np.ones((1,1,300)),sfreq=250)
    tools=EEGTools(d,tmp_path/'out')
    with pytest.raises(ValueError,match='outside permitted'):
        tools.invoke('extract_features',{'source':str(external)})
    with pytest.raises(ValidationError):
        tools.invoke('extract_features',{'source':'$input','arbitrary_python':'rm'})
    step=Plan.model_validate({'steps':[{'id':'E1','tool':'extract_features','args':{'source':'$input'}}]})
    assert step.steps[0].tool=='extract_features'
