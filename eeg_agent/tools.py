"""Validated, allowlisted EEG tools; never execute shell/code from model output."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import csv
import json
import re
import uuid
import re
from .science import review_preprocessing
from .reproducibility import ArtifactCache, sha256_file, provenance, json_write_atomic, environment_info, source_bundle
from typing import Callable

import numpy as np

from .schema import (StrictModel, InspectArgs, PreprocessArgs, SpectrumArgs,
                     ConnectivityArgs, VisualizeArgs, FeatureArgs, QcArgs, IntegrityArgs, BatchArgs, ProposalArgs, QualityReportArgs, MetadataArgs)

SUPPORTED_RAW = {'.edf','.bdf','.set','.vhdr','.fif','.cnt','.npy','.npz','.mat'}

@dataclass
class ToolSpec:
    name: str
    description: str
    args: type[StrictModel]
    callback: Callable
    parallel_safe: bool=False

class EEGTools:
    def __init__(self, data_root: str | Path, output_root: str | Path,
                 *, max_source_mb: int=4096):
        self.data_root = Path(data_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise ValueError(f'Data root directory is missing: {self.data_root}')
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_source_mb = max_source_mb
        self.cache=ArtifactCache(self.output_root)
        self._builtins=set()
        self.registry: dict[str,ToolSpec] = {
            'inspect_eeg': ToolSpec('inspect_eeg','Inspect EEG format, channels, sampling rate, duration and annotations.',InspectArgs,self.inspect_eeg),
            'preprocess_eeg': ToolSpec('preprocess_eeg','Preprocess continuous EEG, export windows and FC to new NPZ; never overwrite source.',PreprocessArgs,self.preprocess_eeg),
            'analyze_spectrum': ToolSpec('analyze_spectrum','Compute Welch PSD and aggregated EEG bandpower of processed NPZ.',SpectrumArgs,self.analyze_spectrum),
            'analyze_connectivity': ToolSpec('analyze_connectivity','Summarize Pearson channel connectivity of processed NPZ.',ConnectivityArgs,self.analyze_connectivity),
            'extract_features': ToolSpec('extract_features',
                'Extract handcrafted per-window/channel EEG features: Gaussian differential entropy (raw and per band), Welch absolute/relative/log bandpower, Hjorth parameters, spectral entropy, RMS, line length and more; export CSV/NPZ.',
                FeatureArgs,self.extract_features),
            'visualize_eeg': ToolSpec('visualize_eeg','Save EEG time/frequency/FC figures and JSON summary of processed NPZ.',VisualizeArgs,self.visualize_eeg),
            'assess_eeg_quality': ToolSpec('assess_eeg_quality','Non-destructive EEG QC: flat channels, extreme amplitude, line noise and non-finite data.',QcArgs,self.assess_eeg_quality),
            'propose_preprocessing': ToolSpec('propose_preprocessing','Propose scientifically cautious preprocessing parameters using QC; never mutate EEG.',ProposalArgs,self.propose_preprocessing),
            'validate_eeg': ToolSpec('validate_eeg','Inspect file checksum, shape, channels, non-finite values and integrity.',IntegrityArgs,self.validate_eeg),
            'batch_preprocess': ToolSpec('batch_preprocess','Dataset-level resumable per-file EEG preprocessing, content-cache and manifest.',BatchArgs,self.batch_preprocess),
            'compare_eeg_quality': ToolSpec('compare_eeg_quality',
                'Read paired before/after EEG preprocessing quality metrics for matching physical channels and time windows; exposes QC report files.',
                QualityReportArgs,self.compare_eeg_quality),
            'inspect_output_metadata': ToolSpec('inspect_output_metadata',
                'Read and validate standardized EEG-Agent output metadata, channel/window shape, sampling, parameters and provenance.',
                MetadataArgs,self.inspect_output_metadata),
        }
        self._builtins=set(self.registry)
        for name in ('inspect_eeg','assess_eeg_quality','validate_eeg','analyze_spectrum','compare_eeg_quality','inspect_output_metadata'):
            self.registry[name].parallel_safe=True

    def register_tool(self, name: str, description: str, args: type[StrictModel], callback: Callable, *, parallel_safe: bool=False):
        """Register host-provided tools via strict Pydantic schemas, no execution of arbitrary code strings."""
        if not isinstance(name,str) or re.fullmatch(r'[a-z][a-z0-9_]{0,63}',name) is None:
            raise ValueError('Invalid tool name')
        if name in self.registry:
            raise ValueError('Tool already registered: '+name)
        if not isinstance(args,type) or not issubclass(args,StrictModel):
            raise TypeError('args must be a StrictModel subclass')
        if not callable(callback) or not isinstance(description,str) or not description.strip():
            raise TypeError('Tool requires a callable and description')
        self.registry[name]=ToolSpec(name,description,args,callback,parallel_safe=parallel_safe)

    def unregister_tool(self,name: str):
        if name in self._builtins:
            raise ValueError('Cannot unregister built-in scientific tools')
        self.registry.pop(name)

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
        source=self._resolve(args.source,suffix=SUPPORTED_RAW)
        params=args.model_dump(exclude={'source','use_cache'})
        review=review_preprocessing(params,input_sfreq=args.input_sfreq)
        if review.errors:
            raise ValueError('; '.join(review.errors))
        # Input hash + scientific parameters + implementation version are part of the key.
        def compute():
            result=self._preprocess_uncached(args)
            result['scientific_warnings']=list(review.warnings)
            return result
        extras=[self._resolve(args.events_tsv,output_allowed=False,suffix={'.tsv'})] if args.events_tsv else []
        for item in source_bundle(source,data_root=self.data_root,extras=extras):
            if item['size_bytes'] > self.max_source_mb*1024*1024:
                raise ValueError('Input EEG companion exceeds source size limit')
        if args.use_cache:
            return self.cache.execute(source,params,'preprocess-v0.4.1',compute,
                                      data_root=self.data_root,extras=extras)
        result=compute()
        artifact=Path(result['output_npz'])
        manifest=artifact.with_suffix('.provenance.json')
        json_write_atomic(manifest,provenance(source,params,algorithm='preprocess-v0.4.1',output=artifact,
                                                   data_root=self.data_root,extras=extras))
        return {**result,'cache_hit':False,'cache_key':None,'provenance_json':str(manifest)}

    def _preprocess_uncached(self,args:PreprocessArgs):
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
            # Paired QC uses the original recording, never the already filtered
            # signal; the retained output windows define matching time intervals.
            from .quality_reports import paired_quality_report
            qc_comparison = paired_quality_report(raw,arrays,meta)
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
        from .quality_reports import (write_quality_report, standardized_output_metadata,
                                      validate_output_metadata)
        qc_files=write_quality_report(qc_comparison,job_dir)
        metadata_path=job_dir/'output_metadata.json'
        std_meta=standardized_output_metadata(
            source=src,output=dest,arrays=arrays,processing_meta=meta,
            params=args.model_dump(exclude={'source','use_cache'}),
            qc_report=qc_comparison,qc_paths=qc_files,data_root=self.data_root,
            artifact_provenance=dest.with_suffix('.provenance.json'))
        validate_output_metadata(std_meta,dest)
        json_write_atomic(metadata_path,std_meta)
        return {'output_npz':str(dest),'shape':meta['shape'],'sfreq':cfg.target_sfreq,
                'window_seconds':meta['effective_window_sec'], 'window_qc':meta['window_qc'],
                'bad_channel_info':meta['bad_channel_info'],'ica':meta['ica'],
                'virtual_interpolated_channels':meta['channel_info']['virtual_interpolated_channels'],
                **qc_files,'qc_comparison_status':qc_comparison['status'],
                'qc_paired_windows':qc_comparison['paired_windows'],
                'output_metadata_json':str(metadata_path)}

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
        from eeg_visualization import read_npz
        from .connectivity import compute_connectivity, DEFAULT_BANDS
        rec=read_npz(src,max_windows=args.max_windows)
        bands=DEFAULT_BANDS if args.bands is None else args.bands
        output=compute_connectivity(rec.data,rec.sfreq,method=args.method,
                                    bands=bands,segment_seconds=args.segment_seconds,backend=args.backend)
        dest=self.output_root/'connectivity'/uuid.uuid4().hex
        dest.mkdir(parents=True,exist_ok=False)
        outfile=dest/'connectivity.npz'
        np.savez_compressed(outfile,**{f'fc_{k}':v for k,v in output['matrices'].items()},
                            channel_names=np.asarray(rec.channels),sfreq=np.asarray(rec.sfreq))
        metrics={}
        for band,mat in output['matrices'].items():
            ix=np.triu_indices(len(rec.channels),k=1)
            vals=np.abs(mat[ix]);metrics[band]={'mean_absolute_connectivity':float(np.mean(vals)),
                                                  'max_absolute_connectivity':float(np.max(vals))}
        info={'source':str(src),'method':args.method,'backend':args.backend,'bands_hz':{k:list(v) for k,v in bands.items()},
              'n_windows_sampled':len(rec.data),'n_channels':rec.n_channels,
              'n_observations':output['n_observations'],'scientific_warnings':output['warnings'],
              'estimator':output['estimator'],'metrics_by_band':metrics,
              'matrix_npz':str(outfile),'note':'Functional connectivity is descriptive, not causal or diagnostic.'}
        if args.method=='pearson':
            m=output['matrices']['broadband'];inds=np.triu_indices(len(rec.channels),1)
            info['mean_absolute_pearson']=float(np.abs(m[inds]).mean())
            edges=sorted(((abs(float(m[i,j])),rec.channels[i],rec.channels[j],float(m[i,j]))
                          for i,j in zip(*inds)),reverse=True)[:10]
            info['strongest_edges']=[{'from':a,'to':b,'pearson_r':v} for _,a,b,v in edges]
        manifest=dest/'connectivity_manifest.json'
        info.update({'source_sha256':sha256_file(src),'environment':environment_info()})
        json_write_atomic(manifest,info)
        info['manifest_json']=str(manifest)
        return info

    def assess_eeg_quality(self,args:QcArgs):
        src=self._resolve(args.source,suffix=SUPPORTED_RAW)
        from .qc import assess_windows
        if src.suffix.lower()=='.npz':
            with np.load(src,allow_pickle=False) as z:
                if 'data' not in z or z['data'].ndim not in (2,3):
                    raise ValueError('QC NPZ must contain EEG data [channels,samples] or [windows,channels,samples].')
                data=z['data'];sfreq=float(np.asarray(z['sfreq']).item())
                if data.ndim==2:data=data[None,:,:]
                unit=str(np.asarray(z['data_unit']).item()) if 'data_unit' in z else 'uV'
                names=list(map(str,z['channel_names'])) if 'channel_names' in z else [f'Ch{i+1}' for i in range(data.shape[1])]
                data=data[:args.max_windows]
        else:
            from eeg_universal import load_raw
            raw=load_raw(src)
            try:
                sfreq=float(raw.info['sfreq']);names=list(raw.ch_names)
                samples=min(raw.n_times,int(sfreq*args.max_raw_seconds))
                data=raw.get_data(start=0,stop=samples)[None,:,:]
                unit='V'
            finally:raw.close()
        result=assess_windows(data,sfreq,unit=unit,flat_uv=args.flat_uv,
            high_amplitude_uv=args.high_amplitude_uv,line_ratio_limit=args.line_ratio_limit)
        for c in result['channels']:c['name']=names[c['index']]
        result['flagged_channels']=[names[i] for i in result['flagged_channel_indices']]
        result['source']=str(src)
        return result

    def propose_preprocessing(self,args:ProposalArgs):
        qc=self.assess_eeg_quality(QcArgs(source=args.source,max_windows=args.qc_max_windows))
        # Suggest, do not automatically alter data or silently discard channels.
        params={'source':args.source,'target_sfreq':min(250.,qc['sfreq']),
                'bandpass_low':.5,'bandpass_high':min(45.,.4*qc['sfreq']),
                'notch_freqs':[f for f in (50.,60.) if f < min(250.,qc['sfreq'])/2],
                'use_ica':False,'interpolate_missing':False}
        review=review_preprocessing(params,input_sfreq=qc['sfreq'])
        return {'source':args.source,'qc':qc,'proposal':params,
                'scientific_review':review.to_dict(),
                'requires_human_review':bool(qc['flagged_channels']),
                'note':'No preprocessing has been performed. Researcher must explicitly choose ICA/interpolation.'}

    def validate_eeg(self,args:IntegrityArgs):
        src=self._resolve(args.source,suffix=SUPPORTED_RAW)
        report={'source':str(src),'size_bytes':src.stat().st_size,'sha256':sha256_file(src),
                'format':src.suffix.lower(),'status':'pass','issues':[]}
        if src.suffix.lower()=='.npz':
            # Check versioned metadata sidecar for Core-generated outputs.
            sidecar=src.parent/'output_metadata.json'
            if sidecar.is_file():
                from .quality_reports import validate_output_metadata
                try:
                    validate_output_metadata(json.loads(sidecar.read_text(encoding='utf-8')),src)
                    report['output_metadata_status']='pass'
                    report['output_metadata_json']=str(sidecar)
                except (ValueError,KeyError,OSError,TypeError) as error:
                    report['output_metadata_status']='invalid'
                    report['status']='requires_review'
                    report['issues'].append(f'Output metadata: {error}')
            with np.load(src,allow_pickle=False) as z:
                if 'data' not in z:raise ValueError('NPZ has no EEG data field')
                x=z['data']
                if x.ndim not in (2,3):raise ValueError('EEG NPZ data must be 2D or 3D')
                if x.size==0:raise ValueError('EEG NPZ data must be nonempty')
                bad=int(np.size(x)-np.count_nonzero(np.isfinite(x)))
                report['shape']=list(x.shape);report['nonfinite_samples']=bad
                if bad:report['status']='requires_review';report['issues'].append(f'{bad} non-finite EEG samples')
                sfreq=float(np.asarray(z['sfreq']).item()) if 'sfreq' in z else 0
                if sfreq<=0:report['status']='requires_review';report['issues'].append('Invalid or missing sfreq')
                report['sfreq']=sfreq
                if 'channel_names' in z and len(z['channel_names'])!=x.shape[-2]:
                    report['status']='requires_review';report['issues'].append('Channel-name count mismatch')
        else:
            report.update(self.inspect_eeg(InspectArgs(source=str(src))))
        return report

    def compare_eeg_quality(self, args: QualityReportArgs):
        src=self._resolve(args.source,suffix={'.npz'})
        # Sidecar is tied to a verified processed artifact, not an arbitrary path.
        self.inspect_output_metadata(MetadataArgs(source=str(src)))
        path=src.parent/'quality_comparison.json'
        if not path.is_file():
            raise FileNotFoundError('No paired QC report; reprocess this recording with Core >=0.4.1')
        report=json.loads(path.read_text(encoding='utf-8'))
        from .quality_reports import QC_REPORT_SCHEMA
        if report.get('schema_id')!=QC_REPORT_SCHEMA:
            raise ValueError('Unsupported QC comparison schema')
        return {'source':str(src),'status':report['status'],
                'paired_windows':report['paired_windows'],
                'paired_channels':report['paired_channels'],
                'window_rejection':report['window_rejection'],
                'summary':report.get('summary',{}),
                'comparability':report['comparability'],
                'quality_comparison_json':str(path),
                'quality_comparison_csv':str(src.parent/'quality_comparison.csv'),
                'quality_comparison_html':str(src.parent/'quality_comparison.html')}

    def inspect_output_metadata(self,args:MetadataArgs):
        src=self._resolve(args.source,suffix={'.npz'})
        path=src.parent/'output_metadata.json'
        if not path.is_file():
            raise FileNotFoundError('No standardized metadata; reprocess this recording with Core >=0.4.1')
        from .quality_reports import validate_output_metadata
        info=json.loads(path.read_text(encoding='utf-8'))
        validate_output_metadata(info,src)
        return {'source':str(src),'status':'pass',
                'schema_id':info['schema_id'], 'recording':info['recording'],
                'data':info['data'], 'preprocessing_parameters':info['preprocessing']['parameters'],
                'quality':info['quality'], 'output_metadata_json':str(path)}

    def batch_preprocess(self,args:BatchArgs):
        from .batch import preprocess_dataset
        return preprocess_dataset(self,**args.model_dump(),progress=self._batch_progress)

    @staticmethod
    def _batch_progress(event:dict):
        try:
            from langgraph.config import get_stream_writer
            get_stream_writer()({'kind':'batch_progress',**event})
        except (ImportError,RuntimeError,LookupError):pass

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
            'source_sha256': sha256_file(src),
            'environment': environment_info(),
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
