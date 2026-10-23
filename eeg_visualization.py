"""EEG NPZ visualization, QC and cross-dataset reports.

Works with both eeg_universal.py and legacy PREDCT/HBN/MODMA/TDBRAIN outputs.
This module NEVER alters source NPZ; figures describe selected windows only.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
# Preserve the Jupyter inline backend when imported from a notebook.
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
from scipy.signal import welch, spectrogram
from scipy.integrate import trapezoid
from scipy.spatial.distance import jensenshannon

BANDS = {'Delta (1–4)': (1, 4), 'Theta (4–8)': (4, 8),
         'Alpha (8–13)': (8, 13), 'Beta (13–30)': (13, 30),
         'Gamma (30–45)': (30, 45)}
PLOTS = ('traces', 'stacked_traces', 'psd', 'fft', 'spectrogram',
         'bandpower', 'bandpower_map', 'bandpower_time',
         'fc_heatmap', 'fc_circle', 'fc_network', 'fc_strength',
         'rms', 'rms_time', 'channel_qc', 'fc_distribution', 'qc_summary')
DEFAULT_PLOTS = ('traces', 'psd', 'spectrogram', 'bandpower', 'bandpower_map',
                 'fc_heatmap', 'fc_circle', 'fc_network', 'fc_strength',
                 'rms', 'rms_time', 'channel_qc', 'qc_summary')


def _scalar(arr, default=None):
    if arr is None:
        return default
    a = np.asarray(arr)
    return str(a.flat[0]) if a.size else default


def _unique_indices(n, limit, preferred=None):
    if limit <= 0:
        raise ValueError('max_windows must be positive')
    indices = np.unique(np.round(np.linspace(0, n - 1, min(n, limit))).astype(int))
    if preferred is not None:
        if preferred < 0 or preferred >= n:
            raise IndexError(f'Window index {preferred} outside 0..{n-1}')
        if preferred not in indices:
            indices[int(np.argmin(abs(indices - preferred)))] = preferred
    return np.unique(np.sort(indices))


@dataclass
class EEGRecord:
    path: Path
    data: np.ndarray  # selected windows x channels x samples, always uV
    sfreq: float
    channels: list[str]
    dataset: str
    subject: str
    times: np.ndarray  # selected windows x [start,end] in seconds
    labels: list[str]
    fc: np.ndarray | None
    metadata: dict
    n_total_windows: int
    indices: np.ndarray
    source_unit: str
    positions_xyz: np.ndarray | None = None

    @property
    def n_channels(self):
        return self.data.shape[1]

    @property
    def duration(self):
        return self.data.shape[-1] / self.sfreq


def read_npz(path: str | Path, max_windows=64, assume_unit='uV', preferred_window=0) -> EEGRecord:
    """Read both generations of NPZ, with finite checks and explicit units.

    Does not run untrusted pickle content. The `data` arrays are loaded in full
    before window subsampling because numpy compressed NPZ has no row access.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        if 'data' not in z:
            raise ValueError(f'{path}: NPZ needs data array')
        x = z['data']
        if x.ndim != 3 or not all(dim > 0 for dim in x.shape):
            raise ValueError(f'{path}: expected (windows,channels,samples), got {x.shape}')
        if x.shape[-1] < 2:
            raise ValueError(f'{path}: EEG window requires >=2 samples')
        n, c, _ = x.shape
        sf = float(np.asarray(z['sfreq']).item()) if 'sfreq' in z else float('nan')
        if not np.isfinite(sf) or sf <= 0:
            raise ValueError(f'{path}: invalid or missing sfreq')
        names = list(np.asarray(z['channel_names']).astype(str)) if 'channel_names' in z else [f'Ch{i+1}' for i in range(c)]
        if len(names) != c or len(set(names)) != len(names):
            raise ValueError(f'{path}: channel_names mismatch or duplicated')
        idx = _unique_indices(n, max_windows, preferred_window)
        unit = _scalar(z['data_unit']) if 'data_unit' in z else assume_unit
        factors = {'uv': 1.0, 'µv': 1.0, 'μv': 1.0, 'mv': 1000.0, 'v': 1e6}
        if not unit or unit.lower() not in factors:
            raise ValueError(f'{path}: unsupported data unit {unit!r}')
        factor = factors[unit.lower()]
        selected = np.asarray(x[idx], dtype=np.float64) * factor
        if not np.all(np.isfinite(selected)):
            raise ValueError(f'{path}: nonfinite EEG data in selected windows')
        if 'window_times_sec' in z:
            tt = np.asarray(z['window_times_sec'], dtype=float)
            if tt.shape != (n, 2):
                raise ValueError(f'{path}: invalid window_times_sec shape {tt.shape}')
            times = tt[idx]
        elif 'window_start_sec' in z and 'window_end_sec' in z:
            times = np.column_stack([z['window_start_sec'], z['window_end_sec']])[idx].astype(float)
        else:
            times = np.column_stack((np.arange(n)*x.shape[-1]/sf,
                                     (np.arange(n)+1)*x.shape[-1]/sf))[idx]
        if 'labels' in z:
            labs = np.asarray(z['labels']).astype(str)
            if len(labs) != n:
                raise ValueError(f'{path}: labels length mismatch')
            labels = list(labs[idx])
        else:
            labels = ['Unknown']*len(idx)
        fc = None
        if 'fc' in z:
            all_fc = z['fc']
            if all_fc.shape == (n,c,c):
                fc = np.asarray(all_fc[idx], dtype=np.float64)
            elif all_fc.shape == (c,c):
                fc = np.asarray(all_fc, dtype=np.float64)[None]
            else:
                warnings.warn(f'{path}: bad fc shape {all_fc.shape}, recomputing from EEG', stacklevel=2)
        positions = None
        if 'channel_positions_xyz' in z:
            candidate = np.asarray(z['channel_positions_xyz'],dtype=float)
            if candidate.shape == (c,3) and np.all(np.isfinite(candidate)):
                positions = candidate
        meta = {}
        if 'metadata_json' in z:
            try:
                meta = json.loads(_scalar(z['metadata_json']))
            except (ValueError, TypeError):
                warnings.warn(f'{path}: malformed metadata_json', stacklevel=2)
        dataset = _scalar(z['dataset_names']) if 'dataset_names' in z else str(meta.get('dataset_name', path.parent.name))
        subject = _scalar(z['subject_ids']) if 'subject_ids' in z else str(meta.get('subject_id', path.stem))
    return EEGRecord(path, selected, sf, names, dataset, subject, times,
                     labels, fc, meta, n, idx, str(unit), positions)


def calc_fc(x):
    if x.ndim != 3:
        raise ValueError('calc_fc requires (windows,channels,samples)')
    x = x - x.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(x*x, axis=-1, keepdims=True))
    x = np.divide(x, norm, out=np.zeros_like(x, dtype=float), where=norm>0)
    fc = np.clip(x @ x.transpose(0,2,1), -1., 1.)
    diag = np.arange(fc.shape[-1]); fc[:,diag,diag] = 0
    return fc


def fc_matrix(record, mode='mean', window_index=0):
    if mode == 'concatenate':
        return calc_fc(record.data.transpose(1,0,2).reshape(1,record.n_channels,-1))[0]
    source = record.fc
    if source is None or not np.all(np.isfinite(source)):
        source = calc_fc(record.data)
    if mode == 'mean':
        m = source.mean(axis=0)
    elif mode == 'single':
        if window_index < 0 or window_index >= record.n_total_windows:
            raise IndexError(f'window_index={window_index} outside 0..{record.n_total_windows-1}')
        ix = np.where(record.indices == window_index)[0]
        if len(ix) == 0:
            raise ValueError('Requested window not loaded; increase --max-windows')
        m = source[min(int(ix[0]),len(source)-1)] if len(source)>1 else source[0]
    else:
        raise ValueError(f'unknown fc mode {mode}')
    m = np.asarray(m,float)
    m = np.nan_to_num(.5*(m+m.T), nan=0, posinf=0, neginf=0)
    m = np.clip(m,-1,1)
    np.fill_diagonal(m,0.)
    return m


def psd_estimate(record):
    f, p = welch(record.data, record.sfreq, axis=-1,
                 nperseg=min(record.data.shape[-1],max(2,round(record.sfreq*2))))
    return f, p


def bandpower(record, relative=False):
    freq, p = psd_estimate(record)
    # Integrated band power; omit bands beyond Nyquist and with no valid bins.
    def integrated(a,b):
        mask = (freq>=a)&(freq<=b)
        if mask.sum() >= 2:
            return trapezoid(p[:,:,mask],freq[mask],axis=-1)
        return np.full(p.shape[:2], np.nan)
    values = {}; valid = {}
    total = integrated(1, min(45.,freq[-1]))
    for name,(a,b) in BANDS.items():
        if freq[-1] < b or (freq>=a).sum() < 2:
            continue
        power = integrated(a,b)
        if np.all(np.isnan(power)):
            continue
        values[name] = np.divide(power,total,out=np.zeros_like(power), where=total>0)*100 if relative else power
        valid[name] = (a,b)
    return values


def _montage_xy(names, positions_xyz=None):
    """Physically meaningful 2D projections from MNE standard montage, not synthetic head locations."""
    if positions_xyz is not None:
        xyz = np.asarray(positions_xyz)
        return {i:xyz[i,:2] for i in range(len(names)) if np.linalg.norm(xyz[i])>1e-10}
    import mne
    coords = {}
    for mn in ('standard_1020','GSN-HydroCel-129'):
        try:
            montage = mne.channels.make_standard_montage(mn)
            coords.update({k.lower(): np.asarray(v)[:2] for k,v in montage.get_positions()['ch_pos'].items()
                           if np.isfinite(v).all()})
        except Exception:
            pass
    return {i:coords[x.lower()] for i,x in enumerate(names) if x.lower() in coords}


def _save(fig, path:Path, formats, dpi, transparent=False):
    out = []
    for fmt in formats:
        name = path.with_suffix('.'+fmt)
        fig.savefig(name, bbox_inches='tight', dpi=dpi, facecolor='none' if transparent else 'white',
                    transparent=transparent)
        out.append(name)
    plt.close(fig)
    return out


def _figure(title, figsize=(9,5.5)):
    fig,ax = plt.subplots(figsize=figsize)
    ax.set_title(title,fontsize=12,pad=13)
    ax.grid(alpha=.18)
    return fig,ax


def _heatmap(values, labels_x, labels_y, title, *, cmap='viridis', vmin=None, vmax=None, label=''):
    side = max(len(labels_x),len(labels_y))
    fig,ax = _figure(title,(max(7,min(15,side*.23+5)),max(5,min(14,side*.23+3))))
    im=ax.imshow(values,interpolation='nearest',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(labels_x))); ax.set_xticklabels(labels_x,rotation=90,fontsize=7 if side>35 else 9)
    ax.set_yticks(range(len(labels_y))); ax.set_yticklabels(labels_y,fontsize=7 if side>35 else 9)
    ax.grid(False)
    fig.colorbar(im,ax=ax,fraction=.045,pad=.02,label=label)
    return fig


def _network(ax, fc, names, max_edges, *, scalp=False, positions_xyz=None):
    c=len(names)
    if c < 2:
        ax.text(.5,.5,'Single channel: no connectivity edges',ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off();return
    positions=_montage_xy(names, positions_xyz) if scalp else {}
    if len(positions)<c:
        angles=np.linspace(np.pi/2,np.pi/2+2*np.pi,c,endpoint=False)
        x=np.cos(angles);y=np.sin(angles)
        projection='circular layout (no complete montage)' if scalp else 'circular layout'
    else:
        xy=np.asarray([positions[i] for i in range(c)])
        # Normalize head XY for stable graph geometry
        xy=xy/(max(np.linalg.norm(xy,axis=1).max(),1e-9))
        x,y=xy[:,0],xy[:,1]; projection='standard montage projection'
    row,col=np.triu_indices(c,k=1)
    order=np.argsort(-np.abs(fc[row,col]))[:max_edges]
    norm=Normalize(-1,1);cmap=plt.get_cmap('RdBu_r')
    for ii in order:
        i,j=int(row[ii]),int(col[ii]); value=float(fc[i,j]);
        ax.plot([x[i],x[j]],[y[i],y[j]],color=cmap(norm(value)),linewidth=.45+2*abs(value),alpha=.3+.6*abs(value),zorder=1)
    ax.scatter(x,y,s=40,color='#223b53',edgecolor='white',zorder=2)
    for i,n in enumerate(names):
        ax.text(x[i]*1.12,y[i]*1.12,n,ha='center',va='center',fontsize=6 if c>=64 else 8)
    ax.set_aspect('equal');ax.set_xlim(-1.3,1.3);ax.set_ylim(-1.3,1.3);ax.axis('off')
    ax.text(.01,.01,projection,transform=ax.transAxes,fontsize=8,color='gray')
    return ScalarMappable(norm=norm,cmap=cmap)


def _fc_circle(ax,fc,names,max_edges,masked_pairs=()):
    c=len(names)
    if c<2:
        ax.text(.5,.5,'Single channel: no connections',ha='center',transform=ax.transAxes)
        ax.axis('off');return
    theta=np.linspace(0,2*np.pi,c,endpoint=False)
    x=np.cos(theta);y=np.sin(theta)
    r,k=np.triu_indices(c,k=1)
    idx=np.argsort(-np.abs(fc[r,k]))[:max_edges]
    colors=plt.get_cmap('RdBu_r');norm=Normalize(-1,1)
    highlighted=set()
    for a,b in masked_pairs:
        if a not in names or b not in names:
            warnings.warn(f'Masked FC connection {a}:{b} absent from this recording',stacklevel=2)
            continue
        i,j=names.index(a),names.index(b)
        if i!=j:highlighted.add(tuple(sorted((i,j))))
    for ix in idx:
        i,j=int(r[ix]),int(k[ix]);value=float(fc[i,j]);
        if (i,j) in highlighted:continue
        vertices=[(x[i],y[i]),(.14*x[i],.14*y[i]),(.14*x[j],.14*y[j]),(x[j],y[j])]
        codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4]
        ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor='none',edgecolor=colors(norm(value)),linewidth=.4+abs(value)*2,alpha=.7))
    for i,j in highlighted:
        vertices=[(x[i],y[i]),(.14*x[i],.14*y[i]),(.14*x[j],.14*y[j]),(x[j],y[j])]
        codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4]
        ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor='none',edgecolor='black',linewidth=3.,zorder=2))
    ax.scatter(x,y,s=42,color='#324e69',edgecolors='white',zorder=3)
    for i,label in enumerate(names):
        ax.text(1.12*x[i],1.12*y[i],label,fontsize=6 if c>=64 else 8,
                ha='center',va='center')
    ax.set_xlim(-1.32,1.32);ax.set_ylim(-1.32,1.32);ax.set_aspect('equal');ax.axis('off')
    return ScalarMappable(norm=norm,cmap=colors)


def _topomap(fig,ax,record,power):
    pos=_montage_xy(record.channels, record.positions_xyz)
    if len(pos)<4:
        ax.text(.5,.5,'Topomap unavailable: fewer than 4 electrodes\nwith known standard montage positions',
                ha='center',va='center',transform=ax.transAxes)
        ax.axis('off');return False
    import mne
    idx=sorted(pos); xy=np.stack([pos[k] for k in idx]); values=np.asarray(power)[idx]
    # scale coordinates; MNE expects points within head sphere
    xy=xy/(max(np.linalg.norm(xy,axis=1).max(),1e-8))*.45
    try:
        im,_=mne.viz.plot_topomap(values,xy,axes=ax,show=False,contours=4,
                                  sensors=True, extrapolate='local')
    except (ValueError,RuntimeError) as exc:
        ax.text(.5,.5,f'Topomap unavailable: {str(exc)[:80]}',ha='center',va='center',transform=ax.transAxes)
        ax.axis('off');return False
    fig.colorbar(im,ax=ax,fraction=.04,label='Relative power (%)')
    return True


def _qc_numbers(rec):
    x=rec.data
    rms=np.sqrt(np.mean(x*x,axis=-1))
    p2p=np.ptp(x,axis=-1)
    meta=rec.metadata.get('window_qc',{})
    rejected=None
    if isinstance(meta,dict):
        for key in ('rejected','rejected_windows','n_rejected','n_rejected_windows'):
            if key in meta and isinstance(meta[key],(int,float)):
                rejected=int(meta[key]); break
    return rms,p2p,rejected


def plot_record(rec: EEGRecord, kind: str, *, mode='mean', window_index=0,
                top_edges=35, max_traces=12, masked_pairs=(),
                absolute_fc=False, diagonal_one=False):
    """Produce one Matplotlib Figure; caller owns saving/closing."""
    if kind not in PLOTS:
        raise ValueError(f'Unknown plot {kind}')
    selected_ix=np.where(rec.indices==window_index)[0]
    ix=int(selected_ix[0]) if len(selected_ix) else 0
    x=rec.data[ix]; time=np.arange(x.shape[-1])/rec.sfreq
    title=f'{rec.dataset} / {rec.subject} · {kind.replace("_"," ")}'
    rms,p2p,rej=_qc_numbers(rec)
    c=rec.n_channels

    if kind in ('traces','stacked_traces'):
        fig,ax=_figure(title, (12,5.8))
        chosen=np.unique(np.linspace(0,c-1,min(c,max_traces)).astype(int))
        waves=x[chosen]
        if kind=='stacked_traces':
            spacing=max(1.,float(np.quantile(np.abs(waves),.98))*2.5)
            for j,k in enumerate(chosen):
                ax.plot(time,waves[j]+j*spacing,lw=.8)
            ax.set_yticks(np.arange(len(chosen))*spacing,labels=[rec.channels[k] for k in chosen]);ax.set_ylabel('Channel / offset')
        else:
            for j,k in enumerate(chosen):ax.plot(time,waves[j],label=rec.channels[k],lw=.9,alpha=.82)
            ax.legend(ncol=min(6,len(chosen)),fontsize=8,loc='upper right')
            ax.set_ylabel('Amplitude (µV)')
        ax.set_xlabel('Time within selected window (s)')
        fig.tight_layout();return fig
    if kind in ('psd','fft'):
        fig,ax=_figure(title)
        f,p=psd_estimate(rec)
        if kind=='psd':
            mean=np.nanmean(p,axis=(0,1)); ax.plot(f,10*np.log10(np.maximum(mean,1e-15)))
            ax.set_ylabel('PSD (dB re 1 µV²/Hz)')
        else:
            yf=np.abs(np.fft.rfft(rec.data,axis=-1)) / rec.data.shape[-1]
            # Convert positive-frequency bins to one-sided amplitude spectrum.
            if rec.data.shape[-1] % 2 == 0:
                yf[...,1:-1] *= 2
            else:
                yf[...,1:] *= 2
            ff=np.fft.rfftfreq(rec.data.shape[-1],1/rec.sfreq)
            ax.plot(ff,np.mean(yf,axis=(0,1))); ax.set_ylabel('Mean single-sided FFT magnitude (µV)')
        ax.set_xlim(0,min(50.,rec.sfreq/2)); ax.set_xlabel('Frequency (Hz)')
        fig.tight_layout();return fig
    if kind=='spectrogram':
        fig,ax=_figure(title,(10,5))
        nseg=min(x.shape[-1],max(8,round(rec.sfreq*.25)))
        freq,t,p=spectrogram(x[0],rec.sfreq,nperseg=nseg,noverlap=int(.75*nseg))
        mask=freq<=min(50.,rec.sfreq/2)
        im=ax.pcolormesh(t,freq[mask],10*np.log10(np.maximum(p[mask],1e-15)),shading='auto',cmap='magma')
        ax.set_xlabel('Time within window (s)');ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title+' · '+rec.channels[0]);fig.colorbar(im,ax=ax,label='PSD (dB re 1 µV²/Hz)')
        fig.tight_layout();return fig
    if kind=='bandpower':
        bp=bandpower(rec,relative=True)
        fig,ax=_figure(title)
        if bp:
            names=list(bp);means=np.array([np.mean(bp[k]) for k in names]);ax.bar(names,means)
            ax.tick_params(axis='x',labelrotation=20);ax.set_ylabel('Relative band power (% of 1–45 Hz)')
        else: ax.text(.5,.5,'No fully resolved frequency bands',ha='center',transform=ax.transAxes)
        fig.tight_layout();return fig
    if kind=='bandpower_map':
        bp=bandpower(rec,relative=True)
        name=next((v for v in bp if v.startswith('Alpha')),next(iter(bp),None))
        fig,ax=_figure(title+(' · '+name if name else ''))
        if name is not None:_topomap(fig,ax,rec,bp[name].mean(axis=0))
        else:ax.text(.5,.5,'No valid bandpower',ha='center',transform=ax.transAxes)
        fig.tight_layout();return fig
    if kind=='bandpower_time':
        bp=bandpower(rec,relative=True)
        if bp:
            bandnames=list(bp); values=np.stack([bp[k].mean(axis=1) for k in bandnames])
            fig=_heatmap(values,[str(i) for i in rec.indices],bandnames,title,label='Relative power (%)')
            fig.axes[0].set_xlabel('Original window index');return fig
        fig,ax=_figure(title);ax.text(.5,.5,'No valid bandpower',transform=ax.transAxes,ha='center');return fig
    if kind in ('fc_heatmap','fc_circle','fc_network','fc_strength','fc_distribution'):
        fc=fc_matrix(rec,mode,window_index)
        if kind=='fc_heatmap':
            fc=fc.copy()
            if diagonal_one:np.fill_diagonal(fc,1.)
            if absolute_fc:fc=np.abs(fc)
            fig=_heatmap(fc,rec.channels,rec.channels,title,cmap='viridis' if absolute_fc else 'RdBu_r',
                         vmin=0 if absolute_fc else -1,vmax=1,label='Absolute Pearson r' if absolute_fc else 'Pearson r')
            return fig
        if kind=='fc_circle':
            fig,ax=_figure(title,(9,8))
            sm=_fc_circle(ax,fc,rec.channels,top_edges,masked_pairs)
            if sm is not None:fig.colorbar(sm,ax=ax,fraction=.035,label='Pearson r')
            return fig
        if kind=='fc_network':
            fig,ax=_figure(title,(9,8))
            sm=_network(ax,fc,rec.channels,top_edges,scalp=True,positions_xyz=rec.positions_xyz)
            if sm is not None:fig.colorbar(sm,ax=ax,fraction=.035,label='Pearson r')
            return fig
        if kind=='fc_strength':
            fig,ax=_figure(title,(max(9,.18*c+6),5.5))
            strength=np.sum(np.abs(fc),axis=1)/max(1,c-1)
            ax.bar(rec.channels,strength);ax.tick_params(axis='x',rotation=90)
            ax.set_ylabel('Mean absolute off-diagonal Pearson r');fig.tight_layout();return fig
        fig,ax=_figure(title);rows,cols=np.triu_indices(c,k=1)
        if len(rows): ax.hist(fc[rows,cols],bins=30,range=(-1,1));ax.set_xlabel('Pearson r (unique channel pairs)')
        else:ax.text(.5,.5,'Single-channel FC has no off-diagonal pairs',ha='center',transform=ax.transAxes)
        ax.set_ylabel('Count');fig.tight_layout();return fig
    if kind=='rms':
        fig,ax=_figure(title)
        ax.hist(rms.ravel(),bins=30);ax.set_xlabel('Window/channel RMS (µV)');ax.set_ylabel('Count')
        fig.tight_layout();return fig
    if kind=='rms_time':
        fig,ax=_figure(title)
        ax.plot(rec.times[:,0],rms.mean(axis=1),marker='o',ms=3)
        ax.set_xlabel('Window start (s)');ax.set_ylabel('Mean channel RMS (µV)');fig.tight_layout();return fig
    if kind=='channel_qc':
        fig,axes=plt.subplots(2,1,figsize=(max(9,.18*c+6),8),sharex=True)
        fig.suptitle(title)
        axes[0].boxplot([rms[:,j] for j in range(c)],showfliers=False);axes[0].set_ylabel('RMS (µV)')
        axes[1].boxplot([p2p[:,j] for j in range(c)],showfliers=False);axes[1].set_ylabel('Peak-to-peak (µV)')
        axes[1].set_xticks(range(1,c+1),rec.channels,rotation=90,fontsize=8)
        for ax in axes:ax.grid(alpha=.2)
        fig.tight_layout();return fig
    if kind=='qc_summary':
        fig,ax=_figure(title,(10,5))
        ax.axis('off')
        meta=rec.metadata; bad=meta.get('bad_channel_info',meta.get('bad_channels',None))
        ica=meta.get('ica',meta.get('ica_excluded_components',None))
        lines=[f'NPZ: {rec.path.name}',f'Dataset / subject: {rec.dataset} / {rec.subject}',
               f'Channels: {c}  |  Sampling: {rec.sfreq:g} Hz',
               f'Window: {rec.duration:.4f} s  ({rec.data.shape[-1]} samples)',
               f'Windows: {rec.n_total_windows} total, {len(rec.indices)} visualized',
               f'Median RMS: {np.median(rms):.3f} µV',
               f'Average peak-to-peak: {np.mean(p2p):.3f} µV',
               f'Explicitly recorded rejected windows: {rej if rej is not None else "not available"}',
               f'Channel repair metadata: {str(bad)[:180]}',
               f'ICA metadata: {str(ica)[:180]}']
        ax.text(.02,.98,'\n\n'.join(lines),va='top',ha='left',transform=ax.transAxes,fontsize=9,linespacing=1.3)
        fig.tight_layout();return fig
    raise AssertionError(kind)


def choose_plots(text):
    if text=='all':return PLOTS
    if text=='default':return DEFAULT_PLOTS
    names=tuple(x.strip() for x in text.split(',') if x.strip())
    wrong=[x for x in names if x not in PLOTS]
    if wrong:raise ValueError(f'Unknown plots {wrong}; options: {", ".join(PLOTS)}')
    return tuple(dict.fromkeys(names))


def describe(rec):
    rms,p2p,rejected=_qc_numbers(rec)
    m=fc_matrix(rec)
    i,j=np.triu_indices(rec.n_channels,1)
    return {'file':str(rec.path),'dataset':rec.dataset,'subject':rec.subject,
            'windows_total':rec.n_total_windows,'windows_sampled':len(rec.indices),
            'channels':rec.n_channels,'samples_per_window':rec.data.shape[-1],
            'window_seconds':rec.duration,'sampling_hz':rec.sfreq,
            'median_rms_uv':float(np.median(rms)), 'mean_p2p_uv':float(np.mean(p2p)),
            'mean_abs_fc':float(np.mean(np.abs(m[i,j]))) if len(i) else float('nan'),
            'rejected_windows_reported':rejected if rejected is not None else '',
            'unit':'uV'}


def _write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _safe_name(value):
    return re.sub(r'[^a-zA-Z0-9._-]+','_',str(value)).strip('._')[:90] or 'record'


def make_report_index(out_dir, entries):
    """Local, self-contained HTML index with relative image links; no JS/network."""
    cards=[]
    for item in entries:
        figs=''.join(f'<a href="{html.escape(str(f))}"><img loading="lazy" src="{html.escape(str(f))}" alt="{html.escape(f.stem)}"><span>{html.escape(f.stem)}</span></a>'
                     for f in item['figures'] if f.suffix=='.png')
        record=item['record']
        cards.append('<section><h2>'+html.escape(record)+' </h2><div class="gallery">'+figs+'</div></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>EEG preprocessing visualization report</title><style>
    body{font:15px system-ui,sans-serif;color:#243244;background:#f6f8fa;max-width:1500px;margin:30px auto;padding:0 24px}
    h1{font-size:26px}section{margin:28px 0;border-top:1px solid #dbe3e9;padding-top:15px}
    .gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
    .gallery a{background:white;color:inherit;text-decoration:none;border:1px solid #e4e7ef;border-radius:9px;padding:8px;box-shadow:0 1px 2px #ddd}
    img{width:100%;height:200px;object-fit:contain}span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px;margin:6px}
    </style><h1>EEG preprocessing visualization report</h1><p>Figures summarize sampled windows. Source NPZ files are unchanged.</p>'''+''.join(cards)+'</html>'
    (out_dir/'index.html').write_text(page,encoding='utf8')


def visualize_file(source, out_dir, *, plots=DEFAULT_PLOTS, max_windows=64,
                   mode='mean', window_index=0, top_edges=35, max_traces=12,
                   formats=('png',),dpi=180, assume_unit='uV',masked_pairs=(),
                   absolute_fc=False,diagonal_one=False,transparent=False):
    rec=read_npz(source,max_windows=max_windows,assume_unit=assume_unit,
                 preferred_window=window_index)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    figures=[];errors={}
    for kind in plots:
        try:
            fig=plot_record(rec,kind,mode=mode,window_index=window_index,
                            top_edges=top_edges,max_traces=max_traces,masked_pairs=masked_pairs,
                            absolute_fc=absolute_fc,diagonal_one=diagonal_one)
            figures += _save(fig,out_dir/kind,formats,dpi,transparent=transparent)
        except (ValueError,RuntimeError,ImportError,IndexError) as exc:
            # Continue reports when a particular visualization is unsupported.
            errors[kind]=str(exc)
            warnings.warn(f'{rec.path.name}: {kind} skipped: {exc}',stacklevel=2)
    summary=describe(rec)
    report={'summary':summary,'requested_plots':list(plots),
            'generated_figures':[p.name for p in figures],'plot_errors':errors,
            'sampling_indices':list(map(int,rec.indices)),
            'note':'Descriptive QC only; selected windows, not clinical diagnosis.'}
    def clean_json(value):
        if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
        if isinstance(value,list):return [clean_json(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
        return value
    (out_dir/'report.json').write_text(json.dumps(clean_json(report),indent=2,ensure_ascii=False,
                                        allow_nan=False,default=str),encoding='utf8')
    return {'record':f'{rec.dataset} / {rec.subject} / {rec.path.name}',
            'figures':figures,'summary':summary,'errors':errors,'source':rec}


def write_validation_tables(items,out_dir):
    """Recreate and extend old notebook's inspectable data checks."""
    out_dir=Path(out_dir)
    validations=[]; metadata=[]
    for item in items:
        rec=item['source']; fc=rec.fc
        if fc is None:
            status='missing (recomputed for charts)'; sym=float('nan'); diag=float('nan'); out_of_range=0
        else:
            status='present';sym=float(np.nanmax(np.abs(fc-np.swapaxes(fc,-2,-1))))
            diag=float(np.nanmax(np.abs(np.diagonal(fc,axis1=-2,axis2=-1))))
            out_of_range=int(np.sum(np.abs(fc)>1+1e-4))
        validations.append({'file':str(rec.path),'fc_status':status,'fc_shape':str(fc.shape) if fc is not None else 'missing',
                            'max_symmetry_error':sym,'max_diagonal_abs':diag,'out_of_range_values':out_of_range,
                            'finite_eeg':bool(np.isfinite(rec.data).all()),'n_channels':rec.n_channels,
                            'n_windows_analyzed':len(rec.indices)})
        meta=rec.metadata
        metadata.append({'file':str(rec.path),'dataset':rec.dataset,'subject':rec.subject,
                         'source_unit':rec.source_unit,'reference':str(meta.get('reference','unknown')),
                         'window_qc':json.dumps(meta.get('window_qc',{}),ensure_ascii=False,default=str),
                         'channel_repair':json.dumps(meta.get('bad_channel_info',meta.get('bad_channels',{})),ensure_ascii=False,default=str),
                         'ICA':json.dumps(meta.get('ica',meta.get('ica_excluded_components',{})),ensure_ascii=False,default=str),
                         'montage_positions': 'stored in NPZ' if rec.positions_xyz is not None else 'standard montage estimate'})
    _write_csv(out_dir/'fc_validation.csv',validations)
    _write_csv(out_dir/'preprocessing_metadata.csv',metadata)


def write_dataset_similarity(records,out_dir,dpi):
    """Descriptive pairwise compatibility checks (not hypothesis tests)."""
    out_dir=Path(out_dir)
    groups={}
    for rec in records:groups.setdefault(rec.dataset,[]).append(rec)
    datasets=sorted(groups)
    names_by={d:set().union(*(set(r.channels) for r in rows)) for d,rows in groups.items()}
    overlaps=[]; overlap_grid=np.zeros((len(datasets),len(datasets)),dtype=float)
    for i,a in enumerate(datasets):
        for j,b in enumerate(datasets):
            intersection=len(names_by[a]&names_by[b]); union=len(names_by[a]|names_by[b])
            val=intersection/union if union else 0
            overlap_grid[i,j]=val
            overlaps.append({'dataset_a':a,'dataset_b':b,'common_electrodes':intersection,
                             'union_electrodes':union,'jaccard':val})
    _write_csv(out_dir/'channel_overlap.csv',overlaps)
    if len(datasets)>1:
        fig=_heatmap(overlap_grid,datasets,datasets,'EEG channel name Jaccard similarity',
                     cmap='Blues',vmin=0,vmax=1,label='Channel Jaccard')
        _save(fig,out_dir/'compare_channel_overlap',('png',),dpi)
    # Each recording contributes equally. Interpolate onto a common frequency grid.
    fmax=min(45.,min(r.sfreq/2 for r in records))
    if fmax<=1:
        (out_dir/'psd_js_divergence_SKIPPED.txt').write_text('Sampling Nyquist frequency <= 1 Hz')
        return
    grid=np.linspace(1,fmax,256)
    by={}
    for key,rows in groups.items():
        p=[]
        for record in rows:
            f,s=psd_estimate(record)
            p.append(np.interp(grid,f,np.mean(s,axis=(0,1))))
        q=np.maximum(np.mean(p,axis=0),0.)
        by[key]=q/max(q.sum(),1e-20)
    js_rows=[];js_matrix=np.zeros_like(overlap_grid)
    for i,a in enumerate(datasets):
        for j,b in enumerate(datasets):
            div=float(jensenshannon(by[a],by[b],base=2.)**2)
            js_rows.append({'dataset_a':a,'dataset_b':b,'psd_js_divergence_bits':div,
                            'frequency_min_hz':1,'frequency_max_hz':fmax,
                            'interpretation':'descriptive, not significance testing'})
            js_matrix[i,j]=div
    _write_csv(out_dir/'psd_js_divergence.csv',js_rows)
    if len(datasets)>1:
        fig=_heatmap(js_matrix,datasets,datasets,'PSD Jensen-Shannon divergence (1–45 Hz, if available)',
                     cmap='magma',vmin=0,vmax=1,label='JS divergence (bits)')
        _save(fig,out_dir/'compare_psd_js',('png',),dpi)


def compare_records(items,out_dir,dpi=180,max_windows=64):
    """File-level dataset descriptions (no pseudo-inferential p-values)."""
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    records=[x['source'] for x in items]
    if len(records)<2:return []
    results=[]
    datasets=sorted(set(r.dataset for r in records))
    summary=[x['summary'] for x in items]
    write_dataset_similarity(records,out_dir,dpi)
    # Equal-file-weight dataset averages; never pretend files are independent people.
    fig,ax=_figure('Dataset comparison · median window RMS per file', (max(8,2*len(datasets)+4),5))
    vals=[[x['median_rms_uv'] for x in summary if x['dataset']==ds] for ds in datasets]
    ax.boxplot(vals,showfliers=True)
    ax.set_xticks(range(1,len(datasets)+1),datasets)
    ax.set_ylabel('Median window RMS (µV)');ax.set_xlabel('Dataset')
    fig.tight_layout();results.extend(_save(fig,out_dir/'compare_rms',('png',),dpi))
    fig,ax=_figure('Dataset comparison · mean PSD of sampled windows')
    for dataset in datasets:
        rows=[]
        for record in records:
            if record.dataset!=dataset:continue
            f,p=psd_estimate(record)
            rows.append((f,p.mean(axis=(0,1))))
        if not rows:continue
        # Different lengths/sfreq -> interpolate onto shared freq axis; retain physical Hz.
        fgrid=np.linspace(1,min(45,min(f[-1] for f,_ in rows)),200)
        avg=np.mean([np.interp(fgrid,f,p) for f,p in rows],axis=0)
        ax.plot(fgrid,10*np.log10(np.maximum(avg,1e-15)),label=f'{dataset} ({len(rows)} files)')
    if ax.lines:ax.legend(fontsize=8)
    ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('PSD (dB re 1 µV²/Hz)')
    fig.tight_layout();results.extend(_save(fig,out_dir/'compare_psd',('png',),dpi))
    fig,ax=_figure('Dataset comparison · relative EEG frequency bands')
    keys=list(BANDS);width=.8/max(1,len(datasets)); x=np.arange(len(keys))
    for i,ds in enumerate(datasets):
        bps=[bandpower(r,relative=True) for r in records if r.dataset==ds]
        values=[np.mean([np.mean(b[k]) for b in bps if k in b]) if any(k in b for b in bps) else np.nan for k in keys]
        ax.bar(x+(i-(len(datasets)-1)/2)*width,values,width,label=ds)
    ax.set_xticks(x,keys,rotation=20);ax.set_ylabel('Relative power (% of 1–45 Hz)');ax.legend(fontsize=8)
    fig.tight_layout();results.extend(_save(fig,out_dir/'compare_bandpower',('png',),dpi))
    # Compare FC ONLY on named intersection, requiring at least 2 channels.
    common=set(records[0].channels)
    for rec in records[1:]:common &= set(rec.channels)
    ordered=[n for n in records[0].channels if n in common]
    if len(ordered)>=2:
        mats=[]
        for ds in datasets:
            part=[]
            for rec in records:
                if rec.dataset!=ds:continue
                idx=[rec.channels.index(name) for name in ordered]
                part.append(fc_matrix(rec)[np.ix_(idx,idx)])
            mats.append(np.mean(part,axis=0))
        fig,axes=plt.subplots(1,len(mats),figsize=(max(6,len(mats)*4),max(4,.19*len(ordered)+3)),squeeze=False)
        for ax,ds,mat in zip(axes[0],datasets,mats):
            im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r')
            ax.set_title(ds); ax.set_xticks(range(len(ordered)),ordered,rotation=90,fontsize=6)
            ax.set_yticks(range(len(ordered)),ordered,fontsize=6)
        fig.suptitle(f'FC comparison · {len(ordered)} identically named channels')
        fig.colorbar(im,ax=axes[0].tolist(),fraction=.025,label='Pearson r')
        results.extend(_save(fig,out_dir/'compare_common_fc',('png',),dpi))
    else:
        (out_dir/'compare_common_fc_SKIPPED.txt').write_text(
            'FC comparison skipped: fewer than 2 identically named channels across all files.\n',encoding='utf8')
    return results


def main(argv=None, preset_plots=None):
    parser=argparse.ArgumentParser(description='Analyze and visualize arbitrary-channel/window EEG NPZ files')
    src=parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--input-file',type=Path,action='append',help='One or more --input-file paths')
    src.add_argument('--input-dir',type=Path)
    src.add_argument('--manifest',type=Path,help='eeg_universal.py manifest.json')
    parser.add_argument('--glob',default='**/*.npz')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plots',default=preset_plots or 'default',help='default / all / comma-separated plot names')
    parser.add_argument('--max-windows',type=int,default=64)
    parser.add_argument('--max-files',type=int,default=100,help='limit for batch processing')
    parser.add_argument('--fc-mode',choices=['mean','single','concatenate'],default='mean')
    parser.add_argument('--window-index',type=int,default=0)
    parser.add_argument('--top-edges',type=int,default=35)
    parser.add_argument('--max-traces',type=int,default=12)
    parser.add_argument('--dpi',type=int,default=180)
    parser.add_argument('--formats',choices=['png','svg','png,svg'],default='png')
    parser.add_argument('--assume-unit',choices=['uV','mV','V'],default='uV',help='for old NPZ with no data_unit')
    parser.add_argument('--compare',action='store_true',help='multi-file descriptive comparison')
    parser.add_argument('--masked-pairs',default='',help='FC circle edges to highlight: Fp1:O2,F3:P4')
    parser.add_argument('--absolute-fc',action='store_true',help='absolute values in FC heatmap only')
    parser.add_argument('--fc-diagonal-one',action='store_true',help='display diagonal as 1 in FC heatmap only')
    parser.add_argument('--transparent',action='store_true',help='transparent figure background')
    args=parser.parse_args(argv)
    try:
        plots=choose_plots(args.plots)
    except ValueError as exc:parser.error(str(exc))
    masked_pairs=[]
    if args.masked_pairs:
        for pair in args.masked_pairs.split(','):
            bits=[v.strip() for v in pair.split(':')]
            if len(bits)!=2 or not all(bits):parser.error('masked pairs must be Ch1:Ch2,Ch3:Ch4')
            masked_pairs.append(tuple(bits))
    if args.max_files<1 or args.max_windows<1 or args.dpi<1 or args.top_edges<0 or args.max_traces<1:
        parser.error('max-files, max-windows, dpi, max-traces must be positive; top-edges >= 0')
    if args.input_file:sources=args.input_file
    elif args.input_dir:sources=sorted(args.input_dir.glob(args.glob))
    else:
        manifest=json.loads(args.manifest.read_text(encoding='utf8'))
        sources=[Path(item['file']) for item in manifest if item.get('status')=='OK' and 'file' in item]
        # Prefer resolved paths; manifest files may be generated in a working directory.
        sources=[p if p.exists() else args.manifest.parent / p for p in sources]
    sources=list(dict.fromkeys(sources))[:args.max_files]
    if not sources:parser.error('No matching NPZ files')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    items=[]; failures=[]
    for index,source in enumerate(sources):
        try:
            subdir=args.output_dir / f'{index+1:03d}_{_safe_name(source.stem)}'
            result=visualize_file(source,subdir,plots=plots,max_windows=args.max_windows,
                                  mode=args.fc_mode,window_index=args.window_index,
                                  top_edges=args.top_edges,max_traces=args.max_traces,
                                  formats=tuple(args.formats.split(',')),dpi=args.dpi,
                                  assume_unit=args.assume_unit,masked_pairs=masked_pairs,
                                  absolute_fc=args.absolute_fc,diagonal_one=args.fc_diagonal_one,
                                  transparent=args.transparent)
            result['figures']=[p.relative_to(args.output_dir) for p in result['figures']]
            items.append(result)
            print(f'OK {source}: {len(result["figures"])} images; {len(result["errors"])} skipped')
        except Exception as exc:
            failures.append({'file':str(source),'error':str(exc)})
            print(f'ERROR {source}: {exc}')
    _write_csv(args.output_dir/'dataset_summary.csv',[r['summary'] for r in items])
    if items:write_validation_tables(items,args.output_dir)
    if args.compare and len(items)>1:
        compare_figs=compare_records(items,args.output_dir,dpi=args.dpi,max_windows=args.max_windows)
        compare_figs += [p for p in (args.output_dir/'compare_channel_overlap.png',args.output_dir/'compare_psd_js.png') if p.exists()]
        if compare_figs:items.insert(0,{'record':'Cross-record / cross-dataset comparisons',
                                       'figures':[p.relative_to(args.output_dir) for p in compare_figs]})
    make_report_index(args.output_dir,items)
    (args.output_dir/'failed_files.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False),encoding='utf8')
    print(f'Report: {args.output_dir / "index.html"}; {len(failures)} failed files')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
