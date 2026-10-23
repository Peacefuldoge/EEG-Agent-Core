"""Acceptance tests: v0.4.1 paired QC, metadata, batch index and sidecar durability."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from eeg_agent import EEGTools
from eeg_agent.quality_reports import OUTPUT_SCHEMA,QC_REPORT_SCHEMA,BATCH_SCHEMA,validate_output_metadata


@pytest.fixture
def demo(tmp_path):
    data=tmp_path/'dataset';data.mkdir()
    sfreq=250
    rng=np.random.default_rng(241)
    times=np.arange(9*sfreq)/sfreq
    x=np.vstack([
        28*np.sin(2*np.pi*10*times)+.5*rng.normal(size=len(times)),
        18*np.sin(2*np.pi*7*times)+.5*rng.normal(size=len(times)),
        11*np.sin(2*np.pi*15*times)+.5*rng.normal(size=len(times)),
    ])
    file=data/'sub-01.npz'
    np.savez(file,data=x.astype('float32'),sfreq=np.asarray(sfreq),
             channel_names=np.asarray(['F3','F4','C3']))
    return EEGTools(data,tmp_path/'results'),file


def test_preprocessed_auto_qc_and_standard_metadata(demo):
    tools,source=demo
    result=tools.invoke('preprocess_eeg',{'source':str(source),'use_ica':False,'target_count':2})
    for key in ('quality_comparison_json','quality_comparison_csv','quality_comparison_html','output_metadata_json'):
        assert Path(result[key]).is_file(),key
    with Path(result['quality_comparison_json']).open(encoding='utf-8') as f:quality=json.load(f)
    assert quality['schema_id']==QC_REPORT_SCHEMA
    assert quality['status']=='compared'
    assert quality['paired_windows']>0
    assert len(quality['paired_channels'])==2
    assert len(quality['per_channel'])==2
    assert quality['window_rejection']['total']>=quality['paired_windows']
    assert quality['comparability']['limitations']
    for row in quality['per_channel']:
        for key in ('median_rms_uv','median_peak_to_peak_uv','flat_window_fraction'):
            assert np.isfinite(row['before'][key])
            assert np.isfinite(row['after'][key])
            assert np.isclose(row['delta_after_minus_before'][key],row['after'][key]-row['before'][key])
    metadata=json.loads(Path(result['output_metadata_json']).read_text(encoding='utf-8'))
    assert metadata['schema_id']==OUTPUT_SCHEMA
    assert metadata['data']['shape']==result['shape']
    assert metadata['data']['data_unit']=='uV'
    assert metadata['data']['dimension_order']==['window','channel','sample']
    assert metadata['data']['sfreq_hz']==250
    assert metadata['preprocessing']['parameters']['target_count']==2
    assert validate_output_metadata(metadata,Path(result['output_npz']))
    assert tools.invoke('inspect_output_metadata',{'source':result['output_npz']})['status']=='pass'
    qc_tool=tools.invoke('compare_eeg_quality',{'source':result['output_npz']})
    assert qc_tool['paired_windows']==quality['paired_windows']
    assert qc_tool['quality_comparison_html']==result['quality_comparison_html']
    assert tools.invoke('validate_eeg',{'source':result['output_npz']})['output_metadata_status']=='pass'
    cached=tools.invoke('preprocess_eeg',{'source':str(source),'use_ica':False,'target_count':2})
    assert cached['cache_hit'] is True


def test_cache_sidecars_fail_closed(demo):
    tools,source=demo
    a=tools.invoke('preprocess_eeg',{'source':str(source),'use_ica':False})
    p=Path(a['quality_comparison_json'])
    p.write_text('tampered',encoding='utf-8')
    b=tools.invoke('preprocess_eeg',{'source':str(source),'use_ica':False})
    assert b['cache_hit'] is False and b['output_npz']!=a['output_npz']
    p=Path(b['output_metadata_json'])
    metadata=json.loads(p.read_text(encoding='utf-8'))
    metadata['data']['channels']=['WRONG']
    p.write_text(json.dumps(metadata),encoding='utf-8')
    integrity=tools.invoke('validate_eeg',{'source':b['output_npz']})
    assert integrity['output_metadata_status']=='invalid'
    assert integrity['status']=='requires_review'
    c=tools.invoke('preprocess_eeg',{'source':str(source),'use_ica':False})
    assert c['cache_hit'] is False and c['output_npz']!=b['output_npz']


def test_batch_has_standardized_index(demo):
    tools,source=demo
    second=source.with_name('sub-02.npz');second.write_bytes(source.read_bytes())
    result=tools.invoke('batch_preprocess',{
        'source_dir':str(source.parent),
        'preprocessing':{'use_ica':False},'max_files':5})
    assert result['completed']==2
    assert Path(result['batch_metadata_json']).is_file()
    index=json.loads(Path(result['batch_metadata_json']).read_text(encoding='utf-8'))
    assert index['schema_id']==BATCH_SCHEMA
    assert index['counts']['completed']==2
    assert len(index['recordings'])==2
    for entry in index['recordings']:
        assert Path(entry['quality_comparison_json']).exists()
        assert Path(entry['output_metadata_json']).exists()
    repeated=tools.invoke('batch_preprocess',{'source_dir':str(source.parent),
        'preprocessing':{'use_ica':False},'max_files':5})
    assert all(item['cache_hit'] for item in repeated['first_files'])


def test_sidecar_validation_rejects_fake_data(demo):
    tools,source=demo
    result=tools.invoke('preprocess_eeg',{'source':str(source),'use_ica':False})
    info=json.loads(Path(result['output_metadata_json']).read_text(encoding='utf-8'))
    info['data']['data_unit']='V'
    with pytest.raises(ValueError,match='unit'):
        validate_output_metadata(info)
    info=json.loads(Path(result['output_metadata_json']).read_text(encoding='utf-8'))
    info['data']['sha256']='0'*64
    with pytest.raises(ValueError,match='SHA256'):
        validate_output_metadata(info,Path(result['output_npz']))


def test_no_common_channels_is_explicit_not_comparable():
    import mne
    from eeg_agent.quality_reports import paired_quality_report
    raw=mne.io.RawArray(np.full((1,500),1e-6),mne.create_info(['F3'],250,['eeg']),verbose=False)
    samples={'data':np.full((1,1,250),1.),'sfreq':np.asarray(250.),
             'channel_names':np.asarray(['P4']),'window_start_sec':np.asarray([0.]),
             'window_end_sec':np.asarray([1.])}
    quality=paired_quality_report(raw,samples,{'window_qc':{'total':1,'kept':1,'rejected':0}})
    assert quality['status']=='not_comparable'
    assert quality['paired_windows']==0
