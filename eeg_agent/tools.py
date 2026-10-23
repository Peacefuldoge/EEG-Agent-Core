"""Validated, allowlisted EEG tools; never execute shell/code from model output."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import uuid
from typing import Callable

import numpy as np

from .schema import (StrictModel, InspectArgs, PreprocessArgs, SpectrumArgs,
                     ConnectivityArgs, VisualizeArgs)

SUPPORTED_RAW = {'.edf','.bdf','.set','.vhdr','.fif','.cnt','.npy','.npz','.mat'}

@dataclass
class ToolSpec:
    name: str
    description: str
    args: type[StrictModel]
    callback: Callable

class EEGTools:
    def __init__(self, data_root: str | Path, output_root: str | Path,
                 *, max_source_mb: int=4096):
        self.data_root = Path(data_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise ValueError(f'Data root directory is missing: {self.data_root}')
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_source_mb = max_source_mb
        self.registry: dict[str,ToolSpec] = {
            'inspect_eeg': ToolSpec('inspect_eeg','Inspect EEG format, channels, sampling rate, duration and annotations.',InspectArgs,self.inspect_eeg),
            'preprocess_eeg': ToolSpec('preprocess_eeg','Preprocess continuous EEG, export windows and FC to new NPZ; never overwrite source.',PreprocessArgs,self.preprocess_eeg),
            'analyze_spectrum': ToolSpec('analyze_spectrum','Compute Welch PSD and aggregated EEG bandpower of processed NPZ.',SpectrumArgs,self.analyze_spectrum),
            'analyze_connectivity': ToolSpec('analyze_connectivity','Summarize Pearson channel connectivity of processed NPZ.',ConnectivityArgs,self.analyze_connectivity),
            'visualize_eeg': ToolSpec('visualize_eeg','Save EEG time/frequency/FC figures and JSON summary of processed NPZ.',VisualizeArgs,self.visualize_eeg),
        }

    def _resolve(self, path: str, *, output_allowed=True, suffix=None) -> Path:
        p = Path(path).expanduser().resolve(strict=True)
        roots = [self.data_root]
        if output_allowed:
            roots.append(self.output_root)
        if not any(p.is_relative_to(root) for root in roots):
            raise ValueError(f'Input outside permitted data/output roots: {p}')
        if not p.is_file():
            raise ValueError('Expected an input file')
        if suffix is not None and p.suffix.lower() not in suffix:
            raise ValueError(f'Unexpected file extension: {p.suffix}')
        if p.stat().st_size > self.max_source_mb * 1024 * 1024:
            raise ValueError(f'Input exceeds configured {self.max_source_mb} MB limit')
        return p

    def invoke(self, name: str, arguments: dict) -> dict:
        if name not in self.registry:
            raise ValueError(f'Unknown tool: {name}')
        spec = self.registry[name]
        parsed = spec.args.model_validate(arguments)
        return spec.callback(parsed)

    def openai_schemas(self) -> list[dict]:
        return [{'type':'function', 'function':{
            'name': spec.name,
            'description':spec.description,
            'parameters':spec.args.model_json_schema(),
        }} for spec in self.registry.values()]

    def inspect_eeg(self, args: InspectArgs):
        path = self._resolve(args.source,suffix=SUPPORTED_RAW)
        if path.suffix.lower() == '.npz':
            with np.load(path,allow_pickle=False) as z:
                if 'data' in z and z['data'].ndim == 3:
                    x = z['data']
                    channels = [str(c) for c in z['channel_names']] if 'channel_names' in z else [f'EEG{i+1}' for i in range(x.shape[1])]
                    return {'source':str(path),'kind':'preprocessed','channels':channels,
                            'n_channels':len(channels),'n_windows':int(x.shape[0]),
                            'samples_per_window':int(x.shape[2]),'sfreq':float(z['sfreq']),
                            'duration_seconds':round(x.shape[0]*x.shape[2]/float(z['sfreq']),2)}
        from eeg_universal import load_raw
        raw = load_raw(path, input_sfreq=args.input_sfreq, input_channels=args.input_channels)
        try:
            return {'source':str(path),'kind':'continuous','channels':list(raw.ch_names),
                    'n_channels':len(raw.ch_names),'sfreq':float(raw.info['sfreq']),
                    'duration_seconds':round(raw.n_times/float(raw.info['sfreq']),3),
                    'annotation_count':len(raw.annotations)}
        finally:
            raw.close()

    def preprocess_eeg(self,args:PreprocessArgs):
        src = self._resolve(args.source,suffix=SUPPORTED_RAW)
        from eeg_universal import Settings, load_raw, preprocess_raw, load_blocks
        cfg = Settings(target_sfreq=args.target_sfreq, window_seconds=args.window_seconds,
                       stride_seconds=args.stride_seconds, bandpass_low=args.bandpass_low,
                       bandpass_high=args.bandpass_high, notch_freqs=tuple(args.notch_freqs),
                       channels=tuple(args.target_channels) if args.target_channels is not None else None,
                       channel_count=args.target_count, interpolate_missing=args.interpolate_missing,
                       use_ica=args.use_ica)
        raw = load_raw(src,input_sfreq=args.input_sfreq,input_channels=args.input_channels)
        try:
            if args.ignore_annotations:
                blocks=[(0.,raw.n_times/raw.info['sfreq'],cfg.default_label)]
            else:
                event_path=self._resolve(args.events_tsv,output_allowed=False,suffix={'.tsv'}) if args.events_tsv else None
                blocks=load_blocks(event_path,raw,cfg.default_label)
            arrays,meta = preprocess_raw(raw,cfg,event_blocks=blocks)
        finally:
            raw.close()
        # A separate run directory ensures no collision and preserves originals.
        job_dir=self.output_root / 'processed' / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=False)
        name=re.sub(r'[^a-zA-Z0-9_-]+','_',src.stem)[:60] or 'eeg'
        dest=job_dir / f'{name}_preprocessed.npz'
        arrays['subject_ids']=np.asarray([src.stem]*len(arrays['data']))
        arrays['source_file']=np.asarray(str(src))
        meta['source_file']=str(src)
        arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False,default=str))
        np.savez_compressed(dest,**arrays)
        return {'output_npz':str(dest),'shape':meta['shape'],'sfreq':cfg.target_sfreq,
                'window_seconds':meta['effective_window_sec'], 'window_qc':meta['window_qc'],
                'bad_channel_info':meta['bad_channel_info'],'ica':meta['ica'],
                'virtual_interpolated_channels':meta['channel_info']['virtual_interpolated_channels']}

    def analyze_spectrum(self,args:SpectrumArgs):
        src=self._resolve(args.source,suffix={'.npz'})
        from eeg_visualization import read_npz,psd_estimate,bandpower
        rec=read_npz(src,max_windows=args.max_windows)
        freqs,power=psd_estimate(rec)
        bands=bandpower(rec)
        # Computed by SciPy; no LLM-generated numerical results.
        stats={k:float(np.nanmean(v)) for k,v in bands.items() if np.isfinite(v).any()}
        return {'source':str(src),'n_windows_sampled':len(rec.data),'channels':rec.n_channels,
                'mean_bandpower_uv2':stats,'peak_psd_frequency_hz':float(freqs[np.argmax(power.mean(axis=(0,1)))]),
                'psd_units':'uV^2/Hz','note':'Descriptive estimates, not clinical interpretation'}

    def analyze_connectivity(self,args:ConnectivityArgs):
        src=self._resolve(args.source,suffix={'.npz'})
        from eeg_visualization import read_npz,fc_matrix
        rec=read_npz(src,max_windows=args.max_windows)
        mat=fc_matrix(rec)
        a,b=np.triu_indices(rec.n_channels,k=1)
        edges=sorted(((abs(float(mat[i,j])),rec.channels[i],rec.channels[j],float(mat[i,j]))
                      for i,j in zip(a,b)),reverse=True)[:min(10,len(a))]
        return {'source':str(src),'n_windows_sampled':len(rec.data),'n_channels':rec.n_channels,
                'mean_absolute_pearson':float(np.mean(np.abs(mat[a,b]))) if len(a) else None,
                'strongest_edges':[{'from':x,'to':y,'pearson_r':v} for _,x,y,v in edges],
                'note':'Pearson correlation does not imply causation or clinical diagnosis'}

    def visualize_eeg(self,args:VisualizeArgs):
        src=self._resolve(args.source,suffix={'.npz'})
        from eeg_visualization import visualize_file,choose_plots,make_report_index
        plots=choose_plots(args.plots)
        dest=self.output_root/'figures'/uuid.uuid4().hex
        dest.mkdir(parents=True,exist_ok=False)
        result=visualize_file(src,dest,plots=plots,max_windows=args.max_windows)
        make_report_index(dest,[{'record':result['record'],
                                     'figures':[p.relative_to(dest) for p in result['figures']]}])
        return {'source':str(src),'report_html':str(dest/'index.html'),
                'summary_json':str(dest/'report.json'),
                'generated_figures':[str(p) for p in result['figures']],
                'skipped_plots':result['errors']}
