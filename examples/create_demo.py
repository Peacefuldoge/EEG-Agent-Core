"""Create synthetic NPZ examples without downloading clinical EEG datasets."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from eeg_universal import Settings, preprocess_raw
import mne


def generate(output: Path=Path('examples/demo_npz')):
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(97)
    layouts=[('PRED_CT',8,1.0),('HBN',12,1.5),('MODMA',16,0.75),('TDBRAIN',6,2.0)]
    names=['Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4',
           'T7','C3','Cz','C4','T8','Pz']
    files=[]
    for dataset,nch,window in layouts:
        sfreq=250.;t=np.arange(10*250)/sfreq
        data=np.stack([5e-6*np.sin(2*np.pi*(8+i%12)*t+(.2*i))+
                       3e-7*rng.normal(size=len(t)) for i in range(nch)])
        raw=mne.io.RawArray(data,mne.create_info(names[:nch],sfreq,ch_types='eeg'),verbose='ERROR')
        arrays,metadata=preprocess_raw(raw,Settings(window_seconds=window,channels=tuple(names[:nch]),
                                               use_ica=False,max_abs_uv=500,max_p2p_uv=700))
        count=len(arrays['data'])
        arrays['subject_ids']=np.asarray([f'synthetic-{dataset}']*count)
        arrays['dataset_names']=np.asarray([dataset]*count)
        path=output / f'synthetic_{dataset}.npz'
        np.savez_compressed(path,**arrays)
        files.append(path)
    print('Generated synthetic demonstration files:')
    for item in files:print(' ',item)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('examples/demo_npz'))
    generate(p.parse_args().output_dir)
