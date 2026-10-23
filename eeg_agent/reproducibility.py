"""Deterministic provenance manifests and collision-resistant file cache."""
from __future__ import annotations
import hashlib, json, os, platform, sys, time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def sha256_file(path: str|Path, *, chunk=1024*1024)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while data:=f.read(chunk):h.update(data)
    return h.hexdigest()


def source_bundle(source: Path, *, data_root: Path|None=None, extras=()):
    """Hash EEG recording and known external file-format companions.

    BrainVision VHDR refers to binary EEG and marker files. EEGLAB SET may
    reference a matching FDT. Caching only the header would be unsafe.
    """
    source=source.resolve(strict=True)
    paths=[source]
    if source.suffix.lower()=='.vhdr':
        header=source.read_bytes()[:256_000].decode('latin-1',errors='replace')
        for row in header.splitlines():
            if '=' in row:
                key,value=[v.strip() for v in row.split('=',1)]
                if key.lower() in ('datafile','markerfile') and value:
                    paths.append((source.parent/value).resolve(strict=True))
    elif source.suffix.lower()=='.set':
        sidecar=source.with_suffix('.fdt')
        if sidecar.exists():paths.append(sidecar.resolve(strict=True))
    paths.extend(Path(extra).resolve(strict=True) for extra in extras if extra)
    root=data_root.resolve() if data_root else None
    for item in paths:
        if not item.is_file() or (root and not item.is_relative_to(root)):
            raise ValueError(f'EEG source companion outside permitted data root: {item}')
    return [{'path':str(item),'size_bytes':item.stat().st_size,
             'sha256':sha256_file(item)} for item in paths]


def json_write_atomic(path: str|Path, data: dict):
    dest=Path(path); dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding='utf-8')
        os.replace(tmp,dest)
    finally:tmp.unlink(missing_ok=True)


def environment_info():
    packages={}
    for name in ('eeg-agent-core','numpy','scipy','mne','pandas','pydantic','mne-connectivity'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return {'python':sys.version.split()[0],'platform':platform.platform(),'packages':packages}


def provenance(source: Path, params:dict, *, algorithm:str, output:Path|None=None, data_root:Path|None=None, extras=()):
    value={'schema_version':1,'algorithm':algorithm,'created_utc':datetime.now(timezone.utc).isoformat(),
           'source_path':str(source),'source_sha256':sha256_file(source),
           'source_bundle':source_bundle(source,data_root=data_root,extras=extras),
           'source_bytes':source.stat().st_size,'parameters':params,'environment':environment_info()}
    if output and output.is_file():
        value.update({'output_path':str(output),'output_sha256':sha256_file(output),'output_bytes':output.stat().st_size})
    return value


@contextmanager
def operation_lock(path: Path, *, timeout=60.0):
    """Cross-process atomic lock with bounded wait and owner diagnostics.

    Stale lock is deliberately NOT automatically removed; operator should
    inspect PID/process and remove after confirming job termination.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();fd=None
    while fd is None:
        try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError:
            if time.monotonic()-start>timeout:
                raise TimeoutError(f'Operation still locked ({path}). Inspect lock owner before removing.')
            time.sleep(.2)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);fd=None
        yield
    finally:
        if fd is not None:os.close(fd)
        path.unlink(missing_ok=True)


class ArtifactCache:
    def __init__(self,root):
        self.root=Path(root)/'.cache'/'operations';self.root.mkdir(parents=True,exist_ok=True)

    def execute(self,source:Path,params:dict,algorithm:str,compute, *, data_root=None, extras=()):
        bundle=source_bundle(source,data_root=data_root,extras=extras)
        source_hash=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()
        key_material={'algorithm':algorithm,'source_bundle':bundle,'params':params}
        key=hashlib.sha256(json.dumps(key_material,sort_keys=True,default=str).encode()).hexdigest()
        meta=self.root/(key+'.json')
        with operation_lock(self.root/(key+'.lock')):
            if meta.exists():
                try:
                    record=json.loads(meta.read_text(encoding='utf-8'))
                    result=record['result']
                    # An artifact's bytes must still match cached provenance.
                    file=Path(result['output_npz'])
                    # Every sidecar matters: a lost/corrupted QC/metadata
                    # file must invalidate the cache, not yield a partial hit.
                    sidecars=record.get('sidecar_sha256',{})
                    sidecars_valid=(bool(sidecars) and all(
                        Path(path).is_file() and sha256_file(path)==digest
                        for path,digest in sidecars.items()))
                    if (file.is_file() and sha256_file(file)==record.get('artifact_sha256')
                        and sidecars_valid):
                        return {**result,'cache_hit':True,'cache_key':key,'provenance_json':record['provenance_json']}
                except (ValueError,KeyError,OSError,TypeError,UnicodeError):pass
            result=compute()
            artifact=Path(result['output_npz'])
            prov=provenance(source,params,algorithm=algorithm,output=artifact,data_root=data_root,extras=extras)
            prov_path=artifact.with_suffix('.provenance.json')
            json_write_atomic(prov_path,prov)
            sidecars={str(p):sha256_file(p) for k in (
                'output_metadata_json','quality_comparison_json',
                'quality_comparison_csv','quality_comparison_html')
                if result.get(k) for p in [Path(result[k])] if p.is_file()}
            sidecars[str(prov_path)]=sha256_file(prov_path)
            json_write_atomic(meta,{'source_sha256':source_hash,'artifact_sha256':prov['output_sha256'],
                                    'sidecar_sha256':sidecars,
                                    'provenance_json':str(prov_path),'result':result})
            return {**result,'cache_hit':False,'cache_key':key,'provenance_json':str(prov_path)}
