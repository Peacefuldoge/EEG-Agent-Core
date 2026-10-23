# EEG-Agent Core 0.4.0 — Scientific workflow upgrade

Core is **UI-free and LLM-free on base installation**. Web UI / CLI live in a separately versioned `eeg-agent-ui` repository. Do not import Chainlit in Core.

## Tools

Tools available under `EEGTools(data_root,output_root).invoke(name,kwargs)`:

- `inspect_eeg`: metadata inspection
- `assess_eeg_quality`: non-mutating flat-channel, excessive amplitude, 50/60 Hz noise and non-finite screening; thresholds are adjustable, qualitative flags require expert review
- `propose_preprocessing`: return suggested conservative parameters and independent QC result, without modifying data
- `validate_eeg`: SHA256, NPZ dimensionality, sampling frequency, finite values, channel-count checks; raw-file metadata check
- `preprocess_eeg`: existing preprocessing with content-addressed cache and reproducibility manifest
- `batch_preprocess`: bounded recursive dataset file discovery, individually cached preprocess, atomic per-file manifest updates and rerunnable failed items
- `analyze_spectrum`: Welch PSD and bandpower
- `analyze_connectivity`: Pearson, coherence `coh`, imaginary coherency `imcoh`, phase locking `plv`, phase lag index `pli`, weighted PLI `wpli` (scientific cautions below)
- `extract_features`: handcrafted per-channel, per-window features
- `visualize_eeg`: figures and HTML reports

### Example

```python
from eeg_agent import EEGTools

tools = EEGTools('/mnt/d/EEG', './agent_runs')
qc = tools.invoke('assess_eeg_quality', {'source':'/mnt/d/EEG/s01.edf'})
pre = tools.invoke('preprocess_eeg', {'source':'/mnt/d/EEG/s01.edf', 'target_sfreq':250,'use_ica':False})
assert pre['output_npz']
fc = tools.invoke('analyze_connectivity', {'source':pre['output_npz'], 'method':'wpli', 'bands':{'alpha':[8,13]}})
print(fc['matrix_npz'], fc['manifest_json'])
```

For the established MNE-Connectivity epoch-based spectral estimator, install
`pip install 'eeg-agent-core[connectivity]'` and pass `backend='mne'` to the
connectivity tool. The optional backend is not needed for Pearson or the
SciPy-native estimator, and was not numerically integration-tested here because
the MNE-Connectivity package was unavailable in the offline test container.

Use raw V / mV / µV units correctly; MNE read_raw returns V. For NPZ, missing `data_unit` defaults to `uV`, so add metadata if input is V. `coh` represents magnitude coherence (not magnitude-squared coherence), `imcoh` is signed imaginary coherency. Spectral measures pool Fourier coefficients across windows/segments and bins **within each band**; not numerically identical to frequency-resolved MNE-Connectivity estimators. These estimators need multiple statistically independent epochs; short windows, overlapping segments and few epochs make estimates unstable. Do not infer directed influence from correlation/coherence. Default windows are 1 second, so Delta results require scrutiny; short-window FFT resolution and spectral leakage are included in warnings.

### Batch file-level resume and cache

```python
result = tools.invoke('batch_preprocess', {
  'source_dir': '/mnt/d/EEG/continuous', 'recursive': True,
  'max_files': 200, 'continue_on_error': True,
  'preprocessing': {'target_sfreq':250, 'window_seconds':1, 'use_ica':False}
})
print(result['manifest_json'],result['completed'],result['failed'])
```

Batch only considers allowed raw extensions, is restricted to `data_root`, and processes **one file at a time**. A file is reprocessed when its source content, companion file (BrainVision EEG/VMRK, EEGLAB FDT), annotated event TSV, parameters, or implementation cache version changes. Outputs are not rewritten when a hash-verified matching artifact exists. A JSON manifest is atomically rewritten after **each individual file**, so restart uses the same call and skips already-complete inputs via the cache. If a raw preprocessing process crashes **mid-file**, that file may be recomputed; batch restart is not intra-file / sample-level checkpointing. Content hashing can take time for multi-gigabyte files. A stale `.lock` requires manual inspection before removal. Storage files should not be concurrently edited manually.

Cache/provenance directories:

```
agent_runs/.cache/operations/<sha256>.json
agent_runs/processed/<run-id>/<record>_preprocessed.npz
agent_runs/processed/<run-id>/<record>_preprocessed.provenance.json
agent_runs/batches/batch_<hash>.json
agent_runs/connectivity/<run-id>/connectivity.npz
agent_runs/connectivity/<run-id>/connectivity_manifest.json
```

Provenance tracks SHA256 for source and linked companion files, artifact hash, full inputs/parameters, versions of scientific dependencies, Python/platform and UTC time. Provenance is intended to aid scientific reproducibility, not prove the algorithm's scientific validity.

### Dynamic extensions and independent steps

```python
from eeg_agent.schema import StrictModel

class MyArgs(StrictModel):
    label: str

tools.register_tool('my_quality_score', 'Read-only user quality index',
                    MyArgs, lambda a: {'value':a.label}, parallel_safe=True)
print(tools.invoke('my_quality_score', {'label':'example'}))
```

Only set `parallel_safe=True` when the callback is **genuinely read-only, thread-safe and independent**. The planner supports dynamic tool names after registry validation; arbitrary tool names/scripts cannot bypass registration. ReWOO and Plan-and-Execute may execute up to three adjacent independent read-only steps in a thread pool, respecting explicit `depends_on`. Write tools are not run in parallel. Their results are merged in deterministic plan order. `qc` preflight executes automatically for all three strategies; planning receives `qc_summary`. When a later explicit QC step flags `requires_review`, Plan-and-Execute and ReWOO attempt an additional plan, subject to `max_replans`; neither automatically repairs rejected data.

### Human parameter editing

Approved tools default to `preprocess_eeg` and `batch_preprocess` (even if invoked by registered third-party Agent). A change sent through `resume_run(thread_id, parameters={'target_sfreq':200})` **does not execute the tool**: it validates the edited parameters and triggers **a new approval interrupt**. The user must then call `resume_run(thread_id, approved=True)` again. Original input path cannot be silently changed by parameter edit. Refusal uses `approved=False`.

### Scientific and operational limits

- QC thresholds are heuristics, not verified clinical artifact classification. A quality flag should be reviewed rather than interpreted as a diagnosis.
- Batch is sequential and has file-level resumability; long-running single-file ICA may still restart after a crash.
- Checkpoint-driven HITL and stream events depend on optional LangGraph, not the basic scientific package.
- Current Core is designed for trusted local use, not an exposed multi-tenant server. Third-party tools execute Python supplied by the host and are trusted-code extensions, not a sandbox.
- High-resolution recordings require sufficient RAM because underlying readers may materialize data. Set `max_source_mb` limits appropriately.
- Validate numerical FC estimates against your preferred MNE-Connectivity pipeline before using results in a manuscript.

### Local verification

```
python -m pytest -q
PYTHONPATH=. python examples/v040_workflow.py
```

Tests include synthetic coherent/independent signals, QC faults, immutable originals, cache hit/invalidation, restart, strict tool schema and offline workflow simulations. Full integration with actual LangGraph/vLLM must run in a networked environment with the optional `agent` requirements installed.
