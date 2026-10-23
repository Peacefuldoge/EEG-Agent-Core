"""New v0.4 scientific algorithms, cache, extension hooks and batch durability."""
from pathlib import Path
import json
import numpy as np
import pytest
from pydantic import Field
from eeg_agent.tools import EEGTools
from eeg_agent.schema import StrictModel
from eeg_agent.connectivity import compute_connectivity
from eeg_agent.qc import assess_windows
from eeg_agent.science import review_preprocessing
from eeg_agent.execution import parse_plan
from eeg_agent.workflow_support import ready_batch,execute_ready

@pytest.fixture
def env(tmp_path):
    data=tmp_path/'data';output=tmp_path/'out';data.mkdir()
    sf=250.;t=np.arange(6*int(sf))/sf
    raw=np.vstack([np.sin(2*np.pi*10*t+i) for i in range(3)])*25
    path=data/'a.npz';np.savez(path,data=raw,sfreq=np.asarray(sf),channel_names=np.asarray(['F3','F4','C3']))
    return EEGTools(data,output),path

def test_qc_flags_flat_and_nonfinite():
    x=np.random.default_rng(42).normal(0,8,(20,3,500))
    x[:,1]=0
    x[3,2,50]=np.nan
    q=assess_windows(x,250.,unit='uV')
    assert q['status']=='requires_review'
    assert 1 in q['flagged_channel_indices']
    assert q['missing_fraction']>0
    assert len(q['channels'])==3
    assert assess_windows(x,250.,unit='µV')['status']=='requires_review'

def test_coherence_and_phase_estimators():
    sf=250;t=np.arange(500)/sf;rng=np.random.default_rng(7)
    x=np.empty((24,3,500))
    for w in range(24):
        phi=rng.uniform(-np.pi,np.pi)
        common=np.sin(2*np.pi*10*t+phi)
        x[w,0]=common+0.2*rng.normal(size=len(t))
        x[w,1]=np.sin(2*np.pi*10*t+phi+.8)+.2*rng.normal(size=len(t))
        x[w,2]=rng.normal(size=len(t))
    for method in ('pearson','coh','imcoh','plv','pli','wpli'):
        y=compute_connectivity(x,sf,method=method,bands={'alpha':(8,13)})
        assert all(np.isfinite(v).all() for v in y['matrices'].values())
        m=next(iter(y['matrices'].values()))
        assert m.shape==(3,3)
        if method in ('coh','plv','pli','wpli'):
            assert m[0,1]>=0
        if method in ('coh','plv'):
            assert m[0,1]>m[0,2]
        if method=='imcoh':assert np.allclose(m,-m.T)

def test_physics_review():
    x=review_preprocessing({'target_sfreq':100,'bandpass_low':.5,'bandpass_high':60})
    assert x.errors
    x=review_preprocessing({'window_seconds':1,'target_sfreq':250})
    assert x.warnings

def test_cache_hit_and_invalidation(env):
    tools,path=env
    opts={'source':str(path),'use_ica':False,'notch_freqs':[50,60]}
    first=tools.invoke('preprocess_eeg',opts)
    second=tools.invoke('preprocess_eeg',opts)
    assert first['cache_hit'] is False
    assert second['cache_hit'] is True
    assert first['output_npz']==second['output_npz']
    prov=json.loads(Path(first['provenance_json']).read_text())
    assert prov['source_sha256'] and prov['output_sha256']
    opts['window_seconds']=2
    third=tools.invoke('preprocess_eeg',opts)
    assert third['cache_hit'] is False and third['output_npz']!=first['output_npz']
    with np.load(path,allow_pickle=False) as z:
        data=z['data']
    np.savez(path,data=data*1.01,sfreq=np.asarray(250.),channel_names=np.asarray(['F3','F4','C3']))
    again=tools.invoke('preprocess_eeg',opts)
    assert again['cache_hit'] is False

def test_batch_manifest_file_level_resume(env):
    tools,path=env
    other=path.with_name('b.npz');other.write_bytes(path.read_bytes())
    opts={'source_dir':str(path.parent),'preprocessing':{'use_ica':False}}
    first=tools.invoke('batch_preprocess',opts)
    assert first['completed']==2 and first['failed']==0
    second=tools.invoke('batch_preprocess',opts)
    assert second['completed']==2
    assert all(row['cache_hit'] for row in second['first_files'])
    assert Path(second['manifest_json']).is_file()
    assert first['job_id']==second['job_id']

def test_registry_is_strict_and_parallel_safe(env):
    tools,path=env
    class Other(StrictModel):
        source:str
        label:str=Field(min_length=1)
    tools.register_tool('own_qa','Custom quality analysis',Other,
                        lambda x:{'tag':x.label},parallel_safe=True)
    assert tools.invoke('own_qa',{'source':str(path),'label':'ok'})['tag']=='ok'
    assert tools.registry['own_qa'].parallel_safe
    with pytest.raises(Exception):tools.invoke('own_qa',{'source':str(path),'label':'ok','danger':'x'})
    plan=parse_plan({'steps':[{'id':'E1','tool':'own_qa','args':{'source':'$input','label':'ok'}}]},allowed_tools=set(tools.registry))
    assert plan[0]['tool']=='own_qa'
    tools.unregister_tool('own_qa')
    with pytest.raises(ValueError):tools.unregister_tool('inspect_eeg')

def test_independent_steps_are_parallel_ready(env):
    tools,path=env
    state={'input_path':str(path),'steps':[
        {'id':'E1','tool':'inspect_eeg','args':{'source':'$input'}},
        {'id':'E2','tool':'validate_eeg','args':{'source':'$input'}},
        {'id':'E3','tool':'assess_eeg_quality','args':{'source':'$input'}},
    ],'evidence':{},'trace':[]}
    batch=ready_batch(state,tools,0)
    assert len(batch)==3
    # Synthetic continuous 2D NPZ not valid QC (tested as explicit failure), but
    # independent successes must be merged in deterministic plan order.
    out=execute_ready(batch,state,tools)
    assert [x['step'] for x in out['trace']]==['E1','E2','E3']
    assert 'E1' in out['evidence']

def test_qc_tool_requires_no_llm(env):
    tools,path=env
    pre=tools.invoke('preprocess_eeg',{'source':str(path),'use_ica':False})
    q=tools.invoke('assess_eeg_quality',{'source':pre['output_npz']})
    assert q['n_channels']==3 and 'status' in q
    proposal=tools.invoke('propose_preprocessing',{'source':pre['output_npz']})
    assert proposal['proposal']['use_ica'] is False
    integrity=tools.invoke('validate_eeg',{'source':pre['output_npz']})
    assert integrity['sha256'] and integrity['status']=='pass'
    fc=tools.invoke('analyze_connectivity',{'source':pre['output_npz'],'method':'coh',
                                            'bands':{'alpha':(8,13)}})
    assert fc['method']=='coh' and Path(fc['matrix_npz']).exists()

def test_companion_source_sha_changes(tmp_path):
    from eeg_agent.reproducibility import source_bundle
    dat=tmp_path/'s.eeg';dat.write_bytes(b'raw-data-v1')
    vmrk=tmp_path/'s.vmrk';vmrk.write_text('[Marker Infos]\n')
    vhdr=tmp_path/'s.vhdr';vhdr.write_text('[Common Infos]\nDataFile=s.eeg\nMarkerFile=s.vmrk\n')
    original=source_bundle(vhdr,data_root=tmp_path)
    assert len(original)==3
    dat.write_bytes(b'raw-data-v2')
    updated=source_bundle(vhdr,data_root=tmp_path)
    assert original[1]['sha256']!=updated[1]['sha256']
    (tmp_path/'outside').mkdir()
    vhdr.write_text('[Common Infos]\nDataFile=../outside.eeg\n')
    with pytest.raises(FileNotFoundError):source_bundle(vhdr,data_root=tmp_path)


def test_corrupt_cached_artifact_recomputes(env):
    tools,path=env
    settings={'source':str(path),'use_ica':False}
    result=tools.invoke('preprocess_eeg',settings)
    target=Path(result['output_npz']);target.write_bytes(b'corrupt')
    second=tools.invoke('preprocess_eeg',settings)
    assert second['cache_hit'] is False
    assert second['output_npz']!=result['output_npz']
