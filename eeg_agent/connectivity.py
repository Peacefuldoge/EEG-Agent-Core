"""Window-aware EEG connectivity: Pearson and cross-spectral estimators.

Coherence, imaginary coherency, PLV, PLI and wPLI are estimated across
windowed Fourier observations, not from a single degenerate FFT. All matrices
are channel-by-channel and descriptive; they do not establish causation.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import stft
from .science import review_connectivity

DEFAULT_BANDS = {'delta':(1.,4.),'theta':(4.,8.),'alpha':(8.,13.),
                 'beta':(13.,30.),'gamma':(30.,45.)}
METHODS = frozenset({'pearson','coh','imcoh','plv','pli','wpli'})


def compute_connectivity(data: np.ndarray, sfreq: float, *, method='coh',
                         bands=None, segment_seconds=1.0, backend='scipy') -> dict:
    x=np.asarray(data,dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError('Expected EEG [windows, channels >= 2, samples].')
    if not np.isfinite(x).all():
        raise ValueError('Connectivity requires finite EEG samples; run QC first.')
    if sfreq <= 0 or method not in METHODS:
        raise ValueError(f'Invalid sampling rate or connectivity method: {method}')
    bands = dict(DEFAULT_BANDS if bands is None else bands)
    review=review_connectivity(method,n_windows=x.shape[0],n_times=x.shape[2],
                               sfreq=sfreq,bands={} if method=='pearson' else bands)
    if review.errors:
        raise ValueError('; '.join(review.errors))
    if backend not in ('scipy','mne'):
        raise ValueError('backend must be scipy or mne')
    n=x.shape[1]
    if method=='pearson':
        corr=[]
        for epoch in x:
            std=np.std(epoch,axis=-1)
            z=(epoch-epoch.mean(axis=-1,keepdims=True))/np.maximum(std[:,None],1e-12)
            mat=z@z.T/epoch.shape[-1]
            mat[std<1e-12,:]=0;mat[:,std<1e-12]=0
            corr.append(mat)
        matrix=np.mean(corr,axis=0)
        np.fill_diagonal(matrix,1.)
        return {'matrices':{'broadband':matrix},'warnings':list(review.warnings),
                'n_observations':len(corr),'method':method,
                'estimator':'mean within-window Pearson correlation'}
    if backend=='mne':
        try:from mne_connectivity import spectral_connectivity_epochs
        except ImportError as exc:
            raise ImportError('Install the optional EEG package: pip install eeg-agent-core[connectivity]') from exc
        if x.shape[0]<2:
            raise ValueError('MNE frequency-domain connectivity requires multiple EEG epochs.')
        matrices={}
        for name,(lo,hi) in bands.items():
            con=spectral_connectivity_epochs(x,sfreq=sfreq,method=method,mode='fourier',
                                             fmin=lo,fmax=hi,faverage=True,verbose=False)
            raw=np.asarray(con.get_data(output='dense'))
            values=raw[...,0] if raw.ndim==3 else raw
            lower=np.tril(values,-1)
            mat=lower-lower.T if method=='imcoh' else lower+lower.T
            if method in ('coh','plv'):np.fill_diagonal(mat,1)
            matrices[name]=mat
        return {'matrices':matrices,'method':method,'warnings':list(review.warnings),
                'n_observations':int(x.shape[0]),
                'frequency_resolution_hz':float(sfreq/x.shape[-1]),
                'estimator':'MNE-Connectivity spectral_connectivity_epochs (Fourier, average across epochs then frequency band)'}
    nperseg=min(x.shape[-1], max(16,int(round(segment_seconds*sfreq))))
    noverlap=nperseg//2 if x.shape[-1]>nperseg else 0
    freqs,_,coeff=stft(x,fs=sfreq,window='hann',nperseg=nperseg,
                       noverlap=noverlap,axis=-1,boundary=None,padded=False)
    # [window, channel, frequency, time-segment]
    matrices={}
    for name,(lo,hi) in bands.items():
        mask=(freqs>=lo)&(freqs<hi)
        if not mask.any():
            raise ValueError(f'{name} contains zero frequency bins at resolution {sfreq/nperseg:.2f} Hz')
        # Samples across windows and STFT segments; pairwise spectral products.
        z=coeff[:,:,mask,:].transpose(1,0,2,3).reshape(n,-1)
        if z.shape[1]<2:
            raise ValueError('At least two Fourier observations required.')
        csd=(z@z.conj().T)/z.shape[1]
        psd=np.maximum(np.real(np.diag(csd)),0.)
        denom=np.sqrt(np.outer(psd,psd))
        coherency=np.divide(csd,denom,out=np.zeros_like(csd),where=denom>1e-24)
        if method=='coh':
            mat=np.abs(coherency)
        elif method=='imcoh':
            mat=np.imag(coherency)
        else:
            mat=np.zeros((n,n),dtype=float)
            for i in range(n):
                for j in range(i):
                    cross=z[i]*z[j].conj()
                    if method=='plv':
                        v=abs(np.mean(np.divide(cross,np.abs(cross),out=np.zeros_like(cross),where=np.abs(cross)>1e-30)))
                    elif method=='pli':
                        v=abs(float(np.mean(np.sign(np.imag(cross)))))
                    else:
                        imag=np.imag(cross)
                        v=float(abs(np.mean(imag))/max(np.mean(abs(imag)),1e-30))
                    mat[i,j]=mat[j,i]=v
        if method=='imcoh':
            # Imaginary coherency is antisymmetric; signed values retain direction of lag.
            np.fill_diagonal(mat,0)
        else:
            mat=(mat+mat.T)/2
            np.fill_diagonal(mat,1 if method in ('coh','plv') else 0)
        matrices[name]=mat
    warnings=list(review.warnings)
    if nperseg==x.shape[-1]:
        warnings.append('One Fourier segment per epoch; estimates are pooled across epochs.')
    return {'matrices':matrices,'method':method,'warnings':warnings,
            'n_observations':int(coeff.shape[0]*coeff.shape[-1]),
            'frequency_resolution_hz':float(sfreq/nperseg),
            'estimator':'pooled cross-spectrum across epochs/STFT segments and band bins'}
