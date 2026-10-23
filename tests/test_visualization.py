"""Offline visualization tests for arbitrary channels and windows."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_visualization import (read_npz, fc_matrix, bandpower, choose_plots,
                               visualize_file, compare_records, main, PLOTS)
from eeg_universal import main as preprocess_cli


def make_npz(path,c=6,samples=325,windows=9,sfreq=250,unit='uV',fc=True,names=None,dataset='EEG-1'):
    t=np.arange(samples)/sfreq
    data=np.stack([np.stack([5*np.sin(2*np.pi*(8+i)*t+0.1*w)
                              for i in range(c)]) for w in range(windows)])
    data=data.astype('float32')
    if unit == 'V':data=data/1e6
    if unit == 'mV':data=data/1e3
    kw=dict(data=data,sfreq=sfreq,channel_names=names or [f'Ch{k}' for k in range(c)],
            subject_ids=np.asarray(['s01']*windows),dataset_names=np.asarray([dataset]*windows),
            window_times_sec=np.stack([np.arange(windows)*samples/sfreq,
                                       (np.arange(windows)+1)*samples/sfreq],axis=1),
            data_unit=unit,labels=['EC']*windows,
            metadata_json=json.dumps({'window_qc':{'n_rejected':3}}))
    if fc:
        kw['fc']=np.stack([np.eye(c,dtype='float32')*.0]*windows)
    np.savez_compressed(path,**kw)
    return path


@pytest.mark.parametrize('channels,samples,sfreq',[(1,100,125),(4,325,250),(26,250,250),(64,750,300)])
def test_variable_channels_windows(tmp_path,channels,samples,sfreq):
    path=make_npz(tmp_path/'x.npz',c=channels,samples=samples,sfreq=sfreq,fc=False)
    rec=read_npz(path,max_windows=5,preferred_window=6)
    assert rec.n_channels==channels
    assert rec.data.shape==(5,channels,samples)
    assert 6 in rec.indices
    assert fc_matrix(rec).shape==(channels,channels)
    assert np.all(np.isfinite(fc_matrix(rec)))
    if samples/sfreq>=1.0 and sfreq>=100: assert bandpower(rec)


def test_unit_conversion_and_no_pickle(tmp_path):
    for unit in ('uV','mV','V'):
        p=make_npz(tmp_path/f'{unit}.npz',unit=unit)
        rec=read_npz(p)
        assert np.max(np.abs(rec.data))==pytest.approx(5,abs=.15)
    p=tmp_path/'bad.npz';np.savez(p,data=np.ones((2,3,1)),sfreq=250)
    with pytest.raises(ValueError,match='2 samples'):read_npz(p)


def test_gallery_and_report_no_fc(tmp_path):
    p=make_npz(tmp_path/'rec.npz',c=4,fc=False,names=['Fp1','Fp2','C3','C4'])
    out=tmp_path/'fig'
    result=visualize_file(p,out,plots=PLOTS,max_windows=8,window_index=7,formats=('png',),dpi=65)
    assert len(result['figures'])==len(PLOTS)
    assert result['errors']=={}
    info=json.loads((out/'report.json').read_text())
    assert info['summary']['window_seconds']==pytest.approx(1.3)
    assert info['summary']['channels']==4
    assert 7 in info['sampling_indices']
    assert (out/'fc_circle.png').is_file()


def test_single_channel_report_and_common_fc_skip(tmp_path):
    a=make_npz(tmp_path/'a.npz',c=1,names=['Fp1'],dataset='A')
    b=make_npz(tmp_path/'b.npz',c=3,names=['Fp2','C3','O2'],dataset='B')
    out=tmp_path/'reports';out.mkdir()
    items=[]
    for p in (a,b):
        res=visualize_file(p,out/p.stem,plots=('fc_network','fc_strength','qc_summary'),dpi=60)
        items.append(res)
    compare_records(items,out,dpi=60)
    assert (out/'compare_common_fc_SKIPPED.txt').is_file()
    assert json.loads((out/'a'/'report.json').read_text())['summary']['mean_abs_fc'] is None


def test_batch_cli_manifest_and_comparison(tmp_path):
    p=make_npz(tmp_path/'a.npz',c=3,dataset='A',names=['Fp1','Fp2','C3'])
    q=make_npz(tmp_path/'b.npz',c=3,dataset='B',names=['Fp1','Fp2','C3'])
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'status':'OK','file':str(p)},{'status':'OK','file':str(q)}]))
    out=tmp_path/'gallery'
    main(['--manifest',str(manifest),'--output-dir',str(out),
          '--plots','psd,fc_heatmap','--max-windows','4','--compare','--dpi','60'])
    assert (out/'index.html').exists()
    assert (out/'compare_common_fc.png').exists()
    assert (out/'dataset_summary.csv').exists()


def test_preprocess_autovisualize(tmp_path):
    from eeg_universal import Settings
    sfreq=250; t=np.arange(8*sfreq)/sfreq
    eeg=np.stack([7*np.sin(2*np.pi*(i+7)*t) for i in range(4)]).astype('float32')
    input_file=tmp_path/'subject.npz'
    np.savez(input_file,data=eeg,sfreq=sfreq,channel_names=['Fp1','Fp2','C3','C4'])
    output_dir=tmp_path/'processed'
    preprocess_cli(['--input-file',str(input_file),'--output-dir',str(output_dir),
                   '--target-count','3','--window-seconds','1.3','--disable-ica',
                   '--visualize','--viz-plots','psd,fc_heatmap','--viz-dpi','55'])
    assert (output_dir/'subject_preprocessed.npz').exists()
    assert (output_dir/'visualizations'/'index.html').exists()
    with np.load(output_dir/'subject_preprocessed.npz') as z:
        assert z['data'].shape[1:] == (3,325)
        assert z['channel_positions_xyz'].shape==(3,3)


def test_bad_plots():
    assert len(choose_plots('all')) >= 16
    with pytest.raises(ValueError, match='Unknown'): choose_plots('magic')


def test_fc_legacy_cli_and_svg_mask(tmp_path):
    p=make_npz(tmp_path/'fc.npz',c=4,names=['Fp1','Fp2','F3','P4'],fc=False)
    out=tmp_path/'circle'
    main(['--input-file',str(p),'--output-dir',str(out),'--plots','fc_circle,fc_heatmap',
          '--fc-mode','single','--window-index','8','--masked-pairs','Fp1:F3',
          '--transparent','--formats','png,svg','--absolute-fc','--fc-diagonal-one','--dpi','65'])
    assert list(out.glob('*/fc_circle.svg'))
    assert list(out.glob('*/fc_heatmap.png'))
    assert (out/'index.html').is_file()


def test_consistency_csv_and_js_divergence(tmp_path):
    a=make_npz(tmp_path/'a.npz',dataset='A',names=['Fp1','Fp2','C3','C4','O1','O2'])
    b=make_npz(tmp_path/'b.npz',dataset='B',names=['Fp1','Fp2','C3','C4','O1','O2'])
    out=tmp_path/'reports'
    main(['--input-file',str(a),'--input-file',str(b),'--output-dir',str(out),
          '--plots','psd','--compare','--dpi','50'])
    for name in ('fc_validation.csv','channel_overlap.csv','psd_js_divergence.csv',
                 'preprocessing_metadata.csv','compare_psd_js.png'):
        assert (out/name).is_file(),name


def test_incorrect_fc_shape_recomputed(tmp_path):
    p=tmp_path/'incorrect.npz'
    np.savez(p,data=np.random.default_rng(0).normal(size=(3,4,250)),sfreq=250,
             fc=np.ones((3,5,5)),channel_names=['Fp1','Fp2','F3','F4'])
    with pytest.warns(UserWarning,match='bad fc shape'):
        rec=read_npz(p)
    assert rec.fc is None
    assert fc_matrix(rec).shape==(4,4)
