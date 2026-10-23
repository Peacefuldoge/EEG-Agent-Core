# Core v0.3.1 — Handcrafted EEG features

- Added `eeg_agent/features.py`: pure NumPy/SciPy multi-channel/multi-window features, no LLM/UI dependency.
- Added `extract_features` as sixth allowlisted `EEGTools` tool; accepts validated Pydantic parameters, obeys data/output path sandbox, exports numeric NPZ + long-format CSV + explanatory JSON.
- Added Gaussian DE (whole-window and five bandpassed windows), Welch PSD absolute/relative/log band powers, Hjorth activity/mobility/complexity, spectral entropy, peak frequency, RMS, PTP, line length and zero-crossing rate.
- Added custom bands, total-band denominator, filtered/PSD variance estimate, Welch settings, user-declared input amplitude units. Unit conversion to uV is explicit.
- Preserves EEG source `labels`, `subject_ids`, `dataset_names`, `run_ids` and `window_times_sec` when present, with correct window subset alignment.
- Public pure-Python API: `from eeg_agent import extract_handcrafted_features`.
- Registered `extract_features` for all three LangGraph strategies and Qwen tool catalog. `features_npz` intentionally doesn't overwrite `latest_npz` reserved for windowed EEG signals.
- Added numerical and synthetic graph tests; existing Core and separate UI 0.3.x architecture retained.
- No changes to `EEG-Agent-UI` are required for Tool Calling; already configured Core 0.3.x dependency range remains compatible.
