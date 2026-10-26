"""File-level resumable preprocessing with per-file atomic manifest updates."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from .reproducibility import json_write_atomic,sha256_file
from .qc_policy import evaluate_tool


def preprocess_dataset(tools, *, source_dir:str, preprocessing:dict,
                       recursive:bool=True,max_files:int=1000,continue_on_error:bool=True,
                       progress=None):
    from .schema import PreprocessArgs
    from .tools import SUPPORTED_RAW
    root=Path(source_dir).expanduser().resolve(strict=True)
    if not root.is_dir() or not root.is_relative_to(tools.data_root):
        raise ValueError('Dataset directory must be inside the permitted data root.')
    all_files=(root.rglob('*') if recursive else root.iterdir())
    files=sorted(p.resolve() for p in all_files if p.is_file() and p.suffix.lower() in SUPPORTED_RAW)
    files=[p for p in files if p.is_relative_to(tools.data_root) and not p.is_relative_to(tools.output_root)]
    if len(files)>max_files:
        raise ValueError(f'{len(files)} files exceed max_files={max_files}; narrow dataset selection.')
    opts={k:v for k,v in preprocessing.items() if k!='source'}
    PreprocessArgs.model_validate({'source':str(files[0]) if files else str(root/'placeholder.edf'),**opts})
    job_key=hashlib.sha256(json.dumps({'root':str(root),'params':opts,'recursive':recursive},sort_keys=True,default=str).encode()).hexdigest()[:24]
    manifest=tools.output_root/'batches'/f'batch_{job_key}.json'
    if manifest.exists():
        state=json.loads(manifest.read_text(encoding='utf-8'))
    else:
        state={'schema_version':1,'job_id':job_key,'dataset_root':str(root),
               'preprocessing':opts,'files':{},'status':'running'}
    file_status=state['files']
    for i,file in enumerate(files):
        # Cache validates checksums and re-computes if a source changed or artifact vanished.
        if progress:progress({'index':i+1,'total':len(files),'file':str(file),'phase':'started'})
        try:
            # Guard each file, not only the first/current session input. Even
            # when a human approved launching this *batch*, files with QC
            # warnings must be handled separately with explicit single-file
            # researcher approval. No user-provided flag bypasses this check.
            try:
                qc=tools.invoke('assess_eeg_quality',{'source':str(file),
                    'max_windows':16,'max_raw_seconds':20,
                    **({'input_unit':opts['input_unit']} if opts.get('input_unit') else {})})
            except Exception as exc:
                qc={'status':'unavailable','error':f'{type(exc).__name__}: {exc}'}
            decision=evaluate_tool('preprocess_eeg',{'source':str(file),**opts},qc)
            qc_record={'status':qc.get('status','unavailable'),
                       'flagged_channels':qc.get('flagged_channels',[]),
                       'reason':decision.reason or qc.get('error','')}
            if decision.action!='allow':
                entry={'status':'requires_review' if decision.action=='review' else 'failed',
                       'source_sha256':sha256_file(file),'qc':qc_record,
                       'error':decision.reason or 'Scientific policy rejected preprocessing'}
                # Never call preprocess_eeg for an unapproved/invalid file.
            else:
                result=tools.invoke('preprocess_eeg',{'source':str(file),**opts})
                entry={'status':'completed','source_sha256':sha256_file(file),
                       'qc':qc_record,
                       'output_npz':result['output_npz'],'provenance_json':result['provenance_json'],
                       'output_metadata_json':result.get('output_metadata_json'),
                       'quality_comparison_json':result.get('quality_comparison_json'),
                       'quality_comparison_csv':result.get('quality_comparison_csv'),
                       'quality_comparison_html':result.get('quality_comparison_html'),
                       'qc_comparison_status':result.get('qc_comparison_status'),
                       'cache_hit':result['cache_hit']}
        except Exception as exc:
            entry={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        file_status[str(file)]=entry
        json_write_atomic(manifest,state)
        if progress:progress({'index':i+1,'total':len(files),'file':str(file),'phase':entry['status']})
        if entry['status']!='completed' and not continue_on_error:break
    counts={'completed':sum(v['status']=='completed' for v in file_status.values()),
            'failed':sum(v['status']=='failed' for v in file_status.values()),
            'requires_review':sum(v['status']=='requires_review' for v in file_status.values())}
    state['status']='completed' if not (counts['failed'] or counts['requires_review']) else 'partial_failure'
    json_write_atomic(manifest,state)
    from datetime import datetime,timezone
    from .quality_reports import BATCH_SCHEMA
    batch_meta=manifest.with_name(manifest.stem+'_metadata.json')
    batch_payload={'schema_id':BATCH_SCHEMA,'schema_version':'1.0',
                   'created_utc':datetime.now(timezone.utc).isoformat(),
                   'job_id':job_key,'dataset_root':str(root),'status':state['status'],
                   'n_files_discovered':len(files),'counts':counts,
                   'preprocessing_parameters':opts,'manifest_json':str(manifest),
                   'recordings':[{'input_path':k,'status':v['status'],
                                  'input_sha256':v.get('source_sha256'),
                                  'output_npz':v.get('output_npz'),
                                  'output_metadata_json':v.get('output_metadata_json'),
                                  'quality_comparison_json':v.get('quality_comparison_json'),
                                  'qc':v.get('qc'),
                                  'error':v.get('error')} for k,v in sorted(file_status.items())]}
    json_write_atomic(batch_meta,batch_payload)
    return {'dataset_root':str(root),'manifest_json':str(manifest),
            'batch_metadata_json':str(batch_meta),'n_files_discovered':len(files),
            'job_id':job_key,'status':state['status'],**counts,
            'first_files':[{ 'source':k,**v} for k,v in list(file_status.items())[:10]],
            'note':'QC is checked on EACH raw file. Batch approval does not override flagged files; review and preprocess those individually. Full results in manifest_json.'}
