"""Validated, allowlisted EEG tools; never execute shell/code from model output."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import csv
import json
import re
import uuid
from typing import Callable

import numpy as np

from .schema import (StrictModel, InspectArgs, PreprocessArgs, SpectrumArgs,
                     ConnectivityArgs, VisualizeArgs, FeatureArgs)

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
            'extract_features': ToolSpec('extract_features',
                'Extract handcrafted per-window/channel EEG features: Gaussian differential entropy (raw and per band), Welch absolute/relative/log bandpower, Hjorth parameters, spectral entropy, RMS, line length and more; export CSV/NPZ.',
                FeatureArgs,self.extract_features),
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

    def extract_features(self, args: FeatureArgs):
        """Produce inspectable, per-window/channel numerical features, not LLM guesses."""
        src = self._resolve(args.source, suffix={'.npz'})
        from .features import read_feature_npz, extract_handcrafted_features
        samples, sfreq, channels, indices, unit, source_meta = read_feature_npz(
            src, max_windows=args.max_windows, assume_unit=args.assume_unit,
            return_metadata=True)
        result = extract_handcrafted_features(
            samples, sfreq, channels=channels, data_unit=unit,
            bands=args.bands, total_band=args.total_band,
            band_de_method=args.band_de_method,
            welch_seconds=args.welch_seconds, filter_order=args.filter_order)

        dest = self.output_root / 'features' / uuid.uuid4().hex
        dest.mkdir(parents=True, exist_ok=False)
        npz_path = dest / 'handcrafted_features.npz'
        csv_path = dest / 'handcrafted_features.csv'
        json_path = dest / 'feature_manifest.json'
        np.savez_compressed(
            npz_path, features=result.values,
            feature_names=np.asarray(result.feature_names),
            feature_units=np.asarray(result.feature_units),
            channel_names=np.asarray(result.channels),
            window_indices=indices, sfreq=np.asarray(sfreq), **source_meta)
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['window_index', 'subject_id', 'dataset', 'label',
                             'window_start_sec', 'window_end_sec', 'channel', *result.feature_names])
            for w, ix in enumerate(indices):
                get = lambda key: str(source_meta[key][w]) if key in source_meta else ''
                times = source_meta['window_times_sec'][w] if 'window_times_sec' in source_meta else ('','')
                for c, name in enumerate(result.channels):
                    writer.writerow([int(ix), get('subject_ids'), get('dataset_names'),
                                     get('labels'), *times, name, *[
                        format(float(x), '.12g') if np.isfinite(x) else ''
                        for x in result.values[w,c]]])

        summary = {}
        for i, name in enumerate(result.feature_names):
            vals = result.values[...,i]
            valid = vals[np.isfinite(vals)]
            summary[name] = {
                'mean': float(valid.mean()) if valid.size else None,
                'std': float(valid.std()) if valid.size else None,
                'invalid_count': int(vals.size - valid.size),
            }
        manifest = {
            'source': str(src),
            'n_windows_sampled': len(indices),
            'window_indices': indices.tolist(),
            'n_channels': len(channels),
            'n_features': len(result.feature_names),
            'shape': list(result.values.shape),
            'sfreq': sfreq,
            'amplitude_unit': 'uV',
            'source_unit': unit,
            'band_de_method': result.band_de_method,
            'relative_power_denominator_hz': list(result.total_band),
            'bands_hz': {k:list(v) for k,v in result.bands.items()},
            'feature_names': result.feature_names,
            'preserved_window_metadata_fields': list(source_meta),
            'feature_units': dict(zip(result.feature_names, result.feature_units)),
            'summary_by_feature': summary,
            'notes': [
                'Gaussian differential entropy uses 0.5*ln(2*pi*e*variance) in nats; amplitude reference is uV.',
                'Relative bandpower is a fraction, not a percentage, of Welch PSD power in the denominator band.',
                'Filtered DE uses per-window forward-backward Butterworth filtering; short-window edge effects apply.',
                'Zero-variance channels yield missing (NaN) entropy instead of fabricated values.',
                'These features are descriptive and not a clinical diagnosis.',
            ],
        }
        json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return {
            'source': str(src),
            'features_npz': str(npz_path),
            'output_csv': str(csv_path),
            'manifest_json': str(json_path),
            'shape': manifest['shape'],
            'n_windows_sampled': manifest['n_windows_sampled'],
            'n_channels': manifest['n_channels'],
            'n_features': manifest['n_features'],
            'feature_names': manifest['feature_names'],
            'relative_power_denominator_hz': manifest['relative_power_denominator_hz'],
            'band_de_method': manifest['band_de_method'],
            'summary_by_feature': summary,
        }

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
