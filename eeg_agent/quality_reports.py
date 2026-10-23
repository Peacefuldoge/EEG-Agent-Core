"""Paired descriptive EEG quality reports and versioned output metadata.

Only compare physically observed, matching named channels in the *same retained*
windows. Retained-only QC is biased by rejection; record the rejected fraction
separately. Never infer clinical quality or that every metric must improve.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np

from .qc import assess_windows
from .reproducibility import environment_info, json_write_atomic, sha256_file

QC_REPORT_SCHEMA = 'org.eeg-agent.qc-comparison/1.0'
OUTPUT_SCHEMA = 'org.eeg-agent.preprocessed-recording/1.0'
BATCH_SCHEMA = 'org.eeg-agent.batch-summary/1.0'


def _finite_stats(data_uv: np.ndarray, sfreq: float, qc_options: dict):
    """Compute transparent descriptive metrics in microvolts, per channel."""
    x = np.asarray(data_uv, dtype=np.float64)
    qc = assess_windows(x, sfreq, unit='uV', **qc_options)
    valid = np.isfinite(x)
    # Invalid samples must be represented separately, never silently discarded.
    cleaned = np.where(valid, x, np.nan)
    if not valid.all():
        with np.errstate(invalid='ignore'):
            # np.nan operations may warn for entirely invalid samples.
            rms = np.sqrt(np.nanmean(cleaned**2, axis=-1))
            ptp = np.nanmax(cleaned, axis=-1) - np.nanmin(cleaned, axis=-1)
    else:
        rms = np.sqrt(np.mean(x*x, axis=-1))
        ptp = np.ptp(x, axis=-1)
    line = qc['line_noise_ratios']
    channels = []
    for index, detail in enumerate(qc['channels']):
        each_rms = rms[:, index]
        each_ptp = ptp[:, index]
        channels.append({
            'flat_window_fraction': detail['flat_window_fraction'],
            'excessive_amplitude_window_fraction': detail['excessive_amplitude_window_fraction'],
            'median_rms_uv': float(np.nanmedian(each_rms)) if np.isfinite(each_rms).any() else None,
            'median_peak_to_peak_uv': float(np.nanmedian(each_ptp)) if np.isfinite(each_ptp).any() else None,
            'line_noise_ratio_50hz': float(line['50'][index]) if '50' in line else None,
            'line_noise_ratio_60hz': float(line['60'][index]) if '60' in line else None,
        })
    return qc, channels


def paired_quality_report(raw, arrays: dict, processing_meta: dict, *, max_windows=64,
                          flat_uv=.25, high_amplitude_uv=250., line_ratio_limit=5.):
    """Compare source and processed data on *matching retained* time intervals.

    Before: unfiltered original MNE Raw in V, converted to uV.
    After: preprocessed output in uV. Channels must share exact name and refer
    to original physical channels. No interpolation is treated as observed.
    """
    out = np.asarray(arrays['data'])
    sfreq_after = float(np.asarray(arrays['sfreq']).item())
    sfreq_before = float(raw.info['sfreq'])
    names_after = [str(c) for c in arrays['channel_names']]
    names_before = list(raw.ch_names)
    excluded = {x['channel'] for x in processing_meta.get('channel_info', {}).get('virtual_interpolated_channels', [])}
    overlap = [c for c in names_after if c in names_before and c not in excluded]
    indices = np.unique(np.linspace(0, len(out)-1, min(len(out), max_windows), dtype=int))
    base = {
        'schema_id': QC_REPORT_SCHEMA,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sampling': 'deterministic_evenly_spaced_retained_windows',
        'retained_windows_total': int(len(out)),
        'paired_windows_requested': int(len(indices)),
        'input_sfreq_hz': sfreq_before,
        'output_sfreq_hz': sfreq_after,
        'original_channels': names_before,
        'processed_channels': names_after,
        'paired_channels': overlap,
        'excluded_synthetic_channels': sorted(excluded),
        'window_rejection': processing_meta.get('window_qc', {}),
        'processing_changes': {
            'filtering_and_notch': True,
            'resampling': not np.isclose(sfreq_before, sfreq_after),
            'referencing': processing_meta.get('reference'),
            'ica': processing_meta.get('ica'),
        },
        'comparability': {
            'time_intervals': 'same retained post-QC windows',
            'channels': 'intersection of exact named physical channels; synthetic channels excluded',
            'sampling_rates': 'metrics calculated at native pre/post sampling rates',
            'amplitude_unit': 'uV',
            'limitations': [
                'Descriptive QC only; a smaller RMS or line-noise ratio does not independently prove improved signal quality.',
                'Only retained output windows are compared; rejected input windows are not included (survivorship bias).',
                'CAR, ICA, filtering and channel selection may change amplitude/distribution independently of artifacts.',
                'When sampling rates differ, metrics are computed on the same time intervals but different sample grids.',
                'Short windows can make 50/60 Hz ratios and flatness estimates unstable.',
            ]
        },
    }
    if not overlap:
        return {**base, 'status': 'not_comparable', 'reason': 'No exact matching observed channel names',
                'paired_windows': 0, 'per_channel': [], 'summary': {}}
    windows_before, windows_after, timings = [], [], []
    ns = max(2, int(round(float(arrays['window_end_sec'][0] - arrays['window_start_sec'][0]) * sfreq_before)))
    for i in indices:
        start = float(arrays['window_start_sec'][i]); end = float(arrays['window_end_sec'][i])
        first = int(round(start*sfreq_before)); last = first+ns
        if first < 0 or last > raw.n_times:
            continue
        segment = raw.get_data(picks=overlap, start=first, stop=last)
        if segment.shape != (len(overlap), ns):
            continue
        windows_before.append(segment * 1e6)
        windows_after.append(out[i, [names_after.index(c) for c in overlap], :])
        timings.append([start, end])
    if not windows_before:
        return {**base, 'status':'not_comparable', 'reason':'No source samples for retained output intervals',
                'paired_windows':0, 'per_channel':[], 'summary':{}}
    options = {'flat_uv':float(flat_uv), 'high_amplitude_uv':float(high_amplitude_uv),
               'line_ratio_limit':float(line_ratio_limit)}
    before_qc, before = _finite_stats(np.stack(windows_before),sfreq_before,options)
    after_qc, after = _finite_stats(np.stack(windows_after),sfreq_after,options)
    names_metrics = tuple(before[0])
    rows=[]
    for name, old, new in zip(overlap,before,after):
        changes={k:(float(new[k]-old[k]) if old[k] is not None and new[k] is not None else None)
                 for k in names_metrics}
        rows.append({'channel':name,'before':old,'after':new,'delta_after_minus_before':changes})
    # Only descriptive mean across same channels (do not mix with rejected fraction).
    metrics={}
    for key in names_metrics:
        valid_pairs=[(r['before'][key],r['after'][key]) for r in rows
                     if r['before'][key] is not None and r['after'][key] is not None]
        metrics[key]={'before_mean':float(np.mean([a for a,_ in valid_pairs])) if valid_pairs else None,
                      'after_mean':float(np.mean([b for _,b in valid_pairs])) if valid_pairs else None,
                      'mean_delta':float(np.mean([b-a for a,b in valid_pairs])) if valid_pairs else None}
    return {**base,'status':'compared','paired_windows':len(windows_before),
            'paired_window_times_sec':timings, 'qc_options':options,
            'before_status':before_qc['status'],'after_status':after_qc['status'],
            'before_missing_fraction':before_qc['missing_fraction'],
            'after_missing_fraction':after_qc['missing_fraction'],
            'per_channel':rows,'summary':metrics}


def write_quality_report(report:dict, target_dir:Path)->dict:
    """Write machine-readable JSON/CSV and a no-JavaScript HTML summary."""
    target_dir=Path(target_dir)
    json_path=target_dir/'quality_comparison.json'
    csv_path=target_dir/'quality_comparison.csv'
    html_path=target_dir/'quality_comparison.html'
    json_write_atomic(json_path,report)
    metrics=tuple(report['per_channel'][0]['before']) if report['per_channel'] else ()
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['channel','metric','before','after','delta_after_minus_before','unit'])
        for row in report['per_channel']:
            for key in metrics:
                unit='uV' if key in ('median_rms_uv','median_peak_to_peak_uv') else 'ratio'
                writer.writerow([row['channel'],key,row['before'][key],row['after'][key],
                                 row['delta_after_minus_before'][key],unit])
    tr=[]
    for row in report['per_channel']:
        for metric in metrics:
            def fmt(val):return '' if val is None else f'{val:.5g}'
            tr.append('<tr>'+''.join(f'<td>{escape(str(value))}</td>' for value in
                (row['channel'],metric,fmt(row['before'][metric]),fmt(row['after'][metric]),
                 fmt(row['delta_after_minus_before'][metric])))+'</tr>')
    notes=''.join(f'<li>{escape(x)}</li>' for x in report['comparability']['limitations'])
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>EEG QC comparison</title>
<style>body{font:15px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#233}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ccd;padding:8px;text-align:left}th{background:#f1f4f6}small{color:#667}</style></head><body>'''
    html+=f'<h1>EEG preprocessing QC comparison</h1><p>Status: {escape(report["status"])} · Paired retained windows: {report["paired_windows"]} · Matched channels: {len(report["paired_channels"])}</p>'
    qc=report.get('window_rejection',{})
    html+=f'<p>Input windows during processing: {qc.get("total", "N/A")}; kept: {qc.get("kept", "N/A")}; rejected: {qc.get("rejected", "N/A")}</p>'
    html+='<table><thead><tr><th>Channel</th><th>Metric</th><th>Before</th><th>After</th><th>Δ after−before</th></tr></thead><tbody>'+''.join(tr)+'</tbody></table>'
    html+='<h2>Interpretation limits</h2><ul>'+notes+'</ul><small>Descriptive QC, not clinical validation.</small></body></html>'
    html_path.write_text(html,encoding='utf-8')
    return {'quality_comparison_json':str(json_path),'quality_comparison_csv':str(csv_path),
            'quality_comparison_html':str(html_path)}


def standardized_output_metadata(*, source:Path, output:Path, arrays:dict, processing_meta:dict,
                                  params:dict, qc_report:dict, qc_paths:dict, data_root:Path,
                                  artifact_provenance:Path)->dict:
    """Versioned sidecar; intentionally NOT presented as BIDS-compliant metadata."""
    x = np.asarray(arrays['data'])
    sf = float(np.asarray(arrays['sfreq']).item())
    ch = [str(s) for s in arrays['channel_names']]
    fields = sorted(arrays)
    times = np.asarray(arrays['window_times_sec'])
    windows = processing_meta.get('window_qc',{})
    return {
        'schema_id':OUTPUT_SCHEMA,'schema_version':'1.0',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'recording':{
            'source_path':str(source),'source_format':source.suffix.lower(),
            'source_sha256':sha256_file(source),'source_bytes':source.stat().st_size,
            'source_relative_to_data_root':str(source.relative_to(data_root))
                if source.is_relative_to(data_root) else None,
            'subject_id':str(source.stem),
        },
        'data':{'path':str(output),'sha256':sha256_file(output),
                'shape':list(map(int,x.shape)),'dimension_order':['window','channel','sample'],
                'dtype':str(x.dtype),'data_unit':'uV','sfreq_hz':sf,
                'window_duration_sec':float(x.shape[2]/sf),
                'channels':ch,'n_channels':len(ch),'n_windows':int(x.shape[0]),
                'window_start_min_sec':float(np.min(times[:,0])),
                'window_end_max_sec':float(np.max(times[:,1])),
                'npz_fields':fields,
                'time_reference':'seconds from the start of the source recording',
                'reference':processing_meta.get('reference'),
                'labels_available':'labels' in arrays},
        'preprocessing':{'parameters':params,'window_qc':windows,
                         'bad_channel_info':processing_meta.get('bad_channel_info'),
                         'ica':processing_meta.get('ica'),
                         'channel_info':processing_meta.get('channel_info')},
        'quality':{'comparison_status':qc_report['status'],
                   'paired_windows':qc_report['paired_windows'],
                   'report_files':qc_paths,'comparison_schema':QC_REPORT_SCHEMA,
                   'interpretation':'Descriptive QC on retained matched windows, not an unbiased estimate of input quality'},
        'provenance':{'path':str(artifact_provenance),'environment':environment_info()},
        'integrity':{'source_sha256':sha256_file(source),'output_sha256':sha256_file(output)},
        'metadata_standard':'EEG-Agent schema; not BIDS sidecar or NWB format',
    }


def validate_output_metadata(payload:dict, output:Path|None=None):
    """Fail closed for mismatched output identity, dimensions, channels or unit."""
    if payload.get('schema_id') != OUTPUT_SCHEMA:
        raise ValueError('Unsupported output metadata schema')
    data=payload.get('data') or {}
    shape=data.get('shape')
    if not isinstance(shape,list) or len(shape)!=3 or any(not isinstance(v,int) or v<1 for v in shape):
        raise ValueError('Invalid EEG shape')
    if data.get('data_unit')!='uV' or data.get('dimension_order')!=['window','channel','sample']:
        raise ValueError('Unsupported EEG units or dimension order')
    if len(data.get('channels',[]))!=shape[1] or data.get('n_channels')!=shape[1] or data.get('n_windows')!=shape[0]:
        raise ValueError('Channel/window counts inconsistent with shape')
    if not isinstance(data.get('sfreq_hz'),(int,float)) or not np.isfinite(data['sfreq_hz']) or data['sfreq_hz']<=0:
        raise ValueError('Invalid sampling frequency')
    if output is not None:
        output=Path(output).resolve(strict=True)
        if Path(data.get('path','')).resolve()!=output or sha256_file(output)!=data.get('sha256'):
            raise ValueError('Output path or SHA256 mismatch')
        with np.load(output,allow_pickle=False) as z:
            if list(z['data'].shape)!=shape or list(map(str,z['channel_names']))!=data['channels']:
                raise ValueError('NPZ contents inconsistent with sidecar')
            if not np.isclose(float(np.asarray(z['sfreq']).item()),data['sfreq_hz']):
                raise ValueError('NPZ sfreq inconsistent with sidecar')
    return True
