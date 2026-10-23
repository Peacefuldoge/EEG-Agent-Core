"""Non-destructive EEG quality assessment; never silently repair source signals."""
from __future__ import annotations
import numpy as np
from scipy.signal import welch


def assess_windows(x: np.ndarray, sfreq: float, *, unit: str='uV',
                   flat_uv:float=.25, high_amplitude_uv:float=250,
                   line_ratio_limit:float=5.) -> dict:
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.size==0 or sfreq<=0:
        raise ValueError('Expected nonempty EEG [windows,channels,samples] and positive sfreq.')
    normalized=str(unit).strip().lower().replace('µ','u').replace('μ','u')
    if normalized not in ('uv','v','mv'):
        raise ValueError('Amplitude unit must be explicitly uV, V or mV.')
    finite=np.isfinite(x)
    missing_fraction=float(1-finite.mean())
    scale={'uv':1.,'mv':1000.,'v':1_000_000.}[normalized]
    clean=np.where(finite,x,0)*scale
    ptp=np.ptp(clean,axis=-1)
    flat=np.nanmean(ptp<flat_uv,axis=0)
    excessive=np.nanmean(np.max(abs(clean),axis=-1)>high_amplitude_uv,axis=0)
    freqs,power=welch(clean,fs=sfreq,axis=-1,nperseg=min(clean.shape[-1],max(16,int(sfreq*2))))
    line={}
    for hz in (50.,60.):
        if hz+3>=sfreq/2:continue
        idx=np.abs(freqs-hz)<=1
        background=(np.abs(freqs-hz)>=2)&(np.abs(freqs-hz)<=6)
        if idx.any() and background.any():
            ratio=np.median(power[...,idx],axis=-1)/(np.median(power[...,background],axis=-1)+1e-20)
            line[str(int(hz))]=np.nanmean(ratio,axis=0).tolist()
    channels=[]
    for c in range(x.shape[1]):
        issues=[]
        if flat[c]>.1:issues.append('flat_or_low_variance')
        if excessive[c]>.1:issues.append('high_amplitude')
        if any(values[c]>line_ratio_limit for values in line.values()):issues.append('line_noise')
        channels.append({'index':c,'flat_window_fraction':round(float(flat[c]),4),
                         'excessive_amplitude_window_fraction':round(float(excessive[c]),4),
                         'issues':issues})
    flagged=[c['index'] for c in channels if c['issues']]
    status='requires_review' if missing_fraction>0 or flagged else 'pass'
    return {'status':status,'n_windows':int(x.shape[0]),'n_channels':int(x.shape[1]),
            'sfreq':float(sfreq),'missing_fraction':missing_fraction,
            'flagged_channel_indices':flagged,'channels':channels,
            'line_noise_ratios':line,
            'recommendations':(['Inspect flagged channels before ICA or interpolation; seek researcher approval.']
                               if flagged else [])+(['Reject or repair non-finite samples before feature computation.'] if missing_fraction else []),
            'notes':['Heuristic screening only; thresholds depend on acquisition system and montage.',
                     'QC does not alter or overwrite the source EEG.']}
