# EEG-Agent Core 0.4.1 — Paired preprocessing QC & versioned output metadata

This extension requires **no UI, LLM or LangGraph installation**. `preprocess_eeg` and `batch_preprocess` transparently create the new QC and metadata artifacts. The two optional read-only Agent tools enable later inspection. Version 0.4.1 invalidates earlier preprocessing cache keys, so **the first run will recompute** recordings.

## Use directly in Python

```python
from eeg_agent import EEGTools

tools = EEGTools(data_root='/mnt/d/EEG', output_root='./agent_runs')
r = tools.invoke('preprocess_eeg', {
    'source': '/mnt/d/EEG/sub-01.edf',
    'target_sfreq': 250,
    'window_seconds': 1,
    'use_ica': False,
})
print(r['output_npz'])
print(r['quality_comparison_html'])   # HTML table with per-channel before/after deltas
print(r['quality_comparison_csv'])    # tidy channel × metric comparison
print(r['quality_comparison_json'])   # full numeric report and interpretability notes
print(r['output_metadata_json'])      # schema-versioned recording metadata

qc = tools.invoke('compare_eeg_quality', {'source': r['output_npz']})
metadata = tools.invoke('inspect_output_metadata', {'source': r['output_npz']})
print(qc['summary'])
print(metadata['data']['shape'])
```

When processing a dataset:

```python
batch = tools.invoke('batch_preprocess', {
    'source_dir': '/mnt/d/EEG/dataset',
    'preprocessing': {'target_sfreq':250, 'window_seconds':1, 'use_ica':False},
})
print(batch['batch_metadata_json'])
```

## Artifacts written for each recording

```
agent_runs/processed/<uuid>/
├── sub-01_preprocessed.npz
├── sub-01_preprocessed.provenance.json
├── quality_comparison.json
├── quality_comparison.csv
├── quality_comparison.html
└── output_metadata.json
```

A batch also writes `agent_runs/batches/batch_<job-id>_metadata.json`: the versioned dataset-level index links every recording to its individual metadata and paired QC report, including failures and per-file cache hits in the operational manifest.

### What is compared?

- **Same intervals**: original recording vs output, at identical retained window time boundaries. Up to **64 retained windows** are chosen deterministically across the output file.
- **Same observed electrodes**: only matching channel names actually present in the original recording. Synthesized/interpolated missing channels are *excluded*. If no physical names match, the report status is `not_comparable`, not "improved".
- **Metrics**: per-channel median RMS (µV), median peak-to-peak (µV), flat-window fraction, excessive-amplitude fraction, and Welch-based 50/60 Hz line-noise/background power ratio (only when Nyquist permits). Deltas are `after − before`; per-channel mean deltas summarize the paired set.
- **Window loss**: the number of windows rejected by preprocessing QC is reported **separately**. Since paired windows are only those retained, comparison of their before/after statistics can suffer survivorship/selection bias. Do not infer a universally better signal based only on a smaller numerical value.
- **Sampling/reference**: original vs processed signals are evaluated at their own sampling frequencies, after resampling/filtering/ICA/CAR when configured. The same time intervals do *not* always contain the same number of samples. Referencing and montage changes affect amplitude metrics. Delta/low-frequency and 50/60-Hz estimates based on short windows are noisy.
- **Units**: `eeg_universal.load_raw` reads MNE Raw in **V**. Numeric NPY/NPZ continuous input uses its loader's **µV default**; confirm source units before calling, especially with third-party datasets.

**Not clinical QC**, no automated diagnosis, and no claim of improved neurophysiological validity. Interpret with raw waveforms, acquisition reference/montage, the number of excluded windows, and a researcher-reviewed methodology.

### Standard metadata schema

`output_metadata.json` has `schema_id=org.eeg-agent.preprocessed-recording/1.0` and carries:

- Source file: path relative to `data_root` if known; SHA256, size, format, subject identifier.
- Output: NPZ path, SHA256, shape `[window,channel,sample]`, signal unit (`uV`), sampling frequency, channel names, time coverage and window metadata fields.
- Processing: exact supplied parameters, window rejection, bad-channel interpolation and ICA summaries.
- Paired QC: report location, comparable window counts and interpretation limits.
- Environment: Python/platform/scientific package versions and pointer to existing source/companion-file provenance manifest.

A machine-readable JSON Schema is packaged as `eeg_agent/resources/output_metadata.schema.json`. `validate_output_metadata(payload, output_npz)` additionally verifies channel names, dimensions, sampling frequency and output SHA256. The `validate_eeg` tool reports a mismatched sidecar as `requires_review`.

**This is EEG-Agent's own versioned metadata format, not a BIDS sidecar or an NWB export.** A future BIDS exporter should create independently BIDS-conformant entities/files and must not simply rename this JSON. For privacy, reports store absolute input paths and subject IDs; review them before external publication.

### Cache and compatibility

The content cache checks SHA256 for NPZ, provenance and all four accompanying QC/metadata files. Missing/modified sidecars cause a **recompute**. Files from versions <0.4.1 do not have these sidecars; the read-only tools fail with an explanation and the user should rerun preprocessing.

UI 0.4.1 can surface the QC HTML/CSV/JSON and metadata JSON from Core tool traces. Its version range remains `eeg-agent-core[agent]>=0.4.0,<0.5.0` so prior 0.4.x combinations remain installable; reading the **new** artifacts requires Core 0.4.1.

## Testing

```
python -m pytest -q
```

Use a suitable EEG dataset to validate heuristic thresholds and artifact behavior prior to scientific publication. These checks do not establish clinical or universal signal quality.
