import json
from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError

from eeg_agent.schema import Plan,PreprocessArgs
from eeg_agent.execution import bind_value,parse_plan,execute_step
from eeg_agent.tools import EEGTools

@pytest.fixture
def sandbox(tmp_path):
    src=tmp_path/'input';src.mkdir()
    out=tmp_path/'output';out.mkdir()
    rng=np.random.default_rng(21)
    sf=200
    t=np.arange(6*sf)/sf
    x=np.stack([10*np.sin(2*np.pi*(7+i*2)*t)+0.3*rng.standard_normal(len(t))
                for i in range(6)]).astype('float64')
    np.savez(src/'synthetic.npz',data=x,sfreq=np.array(sf),
             channel_names=np.array(['F3','F4','C3','C4','P3','P4']))
    return EEGTools(src,out),src/'synthetic.npz'

def test_inspect_preprocess_analyze_and_visualize(sandbox):
    tools,source=sandbox
    observed=tools.invoke('inspect_eeg',{'source':str(source)})
    assert observed['n_channels']==6 and observed['kind']=='continuous'
    result=tools.invoke('preprocess_eeg',{'source':str(source),'target_count':4,
                'target_sfreq':250,'window_seconds':1.3,'stride_seconds':0.7,
                'use_ica':False})
    npz=Path(result['output_npz'])
    assert npz.exists() and npz.is_relative_to(tools.output_root)
    assert result['shape'][1:] == [4,325]
    assert npz != source
    spectrum=tools.invoke('analyze_spectrum',{'source':str(npz)})
    fc=tools.invoke('analyze_connectivity',{'source':str(npz)})
    assert spectrum['channels']==4 and np.isfinite(spectrum['peak_psd_frequency_hz'])
    assert fc['n_channels']==4 and 0<=fc['mean_absolute_pearson']<=1
    vis=tools.invoke('visualize_eeg',{'source':str(npz),'plots':'traces,psd,fc_heatmap'})
    assert Path(vis['report_html']).exists()
    assert len(vis['generated_figures'])==3

def test_path_isolation_and_whitelist(sandbox,tmp_path):
    tools,source=sandbox
    outside=tmp_path/'secret.npy'
    np.save(outside,np.zeros((2,20)))
    with pytest.raises(ValueError,match='outside permitted'):
        tools.invoke('inspect_eeg',{'source':str(outside),'input_sfreq':250})
    with pytest.raises(ValueError,match='Unknown tool'):
        tools.invoke('run_shell',{'command':'echo test'})
    with pytest.raises(ValidationError):
        tools.invoke('inspect_eeg',{'source':str(source),'command':'echo test'})

def test_plan_dag_and_reference_binding(sandbox):
    tools,src=sandbox
    plan={'steps':[
        {'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]},
        {'id':'E2','tool':'preprocess_eeg','args':{'source':'$input','target_count':3,
                    'window_seconds':1.0,'use_ica':False},'depends_on':['E1']},
        {'id':'E3','tool':'analyze_spectrum','args':{'source':'$E2.output_npz'},'depends_on':['E2']},
    ]}
    steps=parse_plan(plan)
    state={'input_path':str(src),'evidence':{},'trace':[]}
    for item in steps:
        outcome=execute_step(Plan.model_validate(plan).steps[steps.index(item)],state,tools)
        state.update(outcome)
        assert state['error'] is None
    assert 'mean_bandpower_uv2' in state['evidence']['E3']
    assert bind_value('$latest_npz',state)==state['evidence']['E2']['output_npz']
    with pytest.raises(ValueError,match='Unsupported reference'):
        bind_value('$E1.__class__.__mro__',state)
    with pytest.raises(ValidationError):
        parse_plan({'steps':[{'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':['E2']}]})
    with pytest.raises(ValidationError):
        parse_plan({'steps':[{'id':'E1','tool':'delete_files','args':{},'depends_on':[]}]})

def test_params_rejected():
    with pytest.raises(ValidationError):
        PreprocessArgs.model_validate({'source':'$input','window_seconds':0})
    with pytest.raises(ValidationError):
        PreprocessArgs.model_validate({'source':'$input','bandpass_high':150,'target_sfreq':250})
    with pytest.raises(ValidationError):
        PreprocessArgs.model_validate({'source':'$input','target_count':6,'target_channels':['F3']})


def test_json_plan_export():
    p=Plan.model_validate({'steps':[{'id':'E1','tool':'inspect_eeg','args':{'source':'$input'}}]})
    assert json.loads(p.model_dump_json())['steps'][0]['depends_on']==[]
