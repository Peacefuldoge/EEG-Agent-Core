# EEG-Agent Core · Handcrafted EEG Features (v0.3.1)

Deterministic NumPy/SciPy extraction; **no LangGraph, Qwen, vLLM, or UI required**. Existing `EEG-Agent-UI` 0.3.x remains compatible without modification. The `extract_features` tool is available to ReAct, Plan-and-Execute, and ReWOO through the existing tool registry.

## Features

For each **EEG window × EEG channel**:

| Feature key | Definition / unit |
|---|---|
| `mean`, `std`, `variance`, `rms`, `peak_to_peak` | uV / uV² |
| `line_length_mean` | Mean absolute adjacent-sample difference, uV/sample |
| `zero_crossing_rate` | Fraction of centered adjacent samples crossing zero |
| `de_gaussian` | Gaussian differential entropy of raw window, nats (uV reference) |
| `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity` | Standard Hjorth statistics, mobility in 1/s |
| `peak_frequency` | Welch PSD peak frequency within total band, Hz |
| `spectral_entropy` | Shannon entropy of normalized Welch spectral bins, divided by log(number of bins) |
| `total_power` | Integral of PSD across denominator band, uV² |
| `power_<band>` | Integrated Welch PSD band power, uV² |
| `relative_power_<band>` | `power_<band> / total_power`, **fraction** (0–1, not percent) |
| `log_power_<band>` | Natural log of band power in uV² |
| `de_<band>` | Gaussian differential entropy of bandpassed EEG variance, nats (uV reference) |

Default band definitions: delta 1–4 Hz; theta 4–8 Hz; alpha 8–13 Hz; beta 13–30 Hz; gamma 30–45 Hz. Default denominator: **1–45 Hz**, not the sum of the five bands and not the full Nyquist interval. Any non-overlapping/overlapping user-defined named bands within the denominator are accepted; overlapping bands mean the relative fractions need not sum to 1. Data must have a sufficiently high sample rate (`Nyquist > 45 Hz` for defaults).

Number of features using five bands: **14 + (5 × 4) = 34**, output `(windows, channels, 34)`. Window selection defaults to at most 32 evenly spaced windows; set `max_windows` higher for large runs (upper API limit 512). Input is loaded as an NPZ with `data` of shape `(windows,channels,samples)`, scalar `sfreq`, optional `channel_names`, and optional `data_unit`. Original EEG is unchanged.

## The DE calculation matters

With the **Gaussian assumption**, differential entropy for a continuous random variable is:

\[
h(X) = \frac12 \ln(2\pi e\sigma_X^2).
\]

**DE is not discrete Shannon entropy**, and its numerical value depends on amplitude unit. All channels are converted to **uV** before extraction, so DE is reported as "nats (uV reference)". If the signal variance is zero, this degenerate Gaussian has no finite DE; the result is `NaN`, with an invalid count in the manifest, not an artificial number.

- `band_de_method="filtered"` (**default**): fourth-order Butterworth SOS bandpass, forward-backward per EEG window; DE uses variance of the **filtered time-domain samples**. Short windows insufficient for `sosfiltfilt` are rejected with a useful error. **This can have serious boundary artifacts for short windows and very low frequencies**; for research-grade band DE, ideally bandpass the *continuous EEG* first, then window it, or use longer windows with sensitivity analysis.
- `band_de_method="psd"`: compute DE formula from *integrated Welch bandpower*, used as a **variance proxy**; not identical to the filtered-variance DE. Useful for very short windows, but the approximation and short-window spectral resolution must be disclosed. It is never silently substituted for the default method.

Welch segment length defaults to 2 seconds but is capped at the window duration; PSD integration interpolates exact frequency band boundaries to reduce dependence on FFT bin alignment. A 1-second window gives coarse resolution at 1–4 Hz (delta), even though the code can produce a number.

## Installation

From the Core source folder, in the independent `eeg_agent` Conda environment:

```bash
conda activate eeg_agent
uv pip install -e . --python "$CONDA_PREFIX/bin/python"
```

Only NumPy, SciPy, and existing scientific Core packages are needed for feature extraction. **No model, LangGraph, Chainlit or CLI must be installed**. Installing `.[agent]` is optional for graph execution.

## 1. Pure Python / NumPy

```python
import numpy as np
from eeg_agent import extract_handcrafted_features

# x: arbitrary windows × channels × samples; example unit is microvolts
x = np.random.default_rng(7).normal(0, 10, size=(4, 26, 250))
f = extract_handcrafted_features(x, sfreq=250, data_unit="uV")
print(f.values.shape)  # (4, 26, 34)
print(f.feature_names)
alpha_relative = f.values[..., f.feature_names.index("relative_power_alpha")]
alpha_de = f.values[..., f.feature_names.index("de_alpha")]
```

For input EEG expressed in volts (as is usual in MNE), pass `data_unit="V"`, *not* `"uV"`.

Custom frequency bands:

```python
f = extract_handcrafted_features(
    x, sfreq=250, bands={"low_alpha": (8, 10), "high_alpha": (10, 13)},
    total_band=(1, 45), band_de_method="filtered"
)
```

## 2. Core Tool API (actual feature extraction + exports)

```python
from eeg_agent import EEGTools

tools = EEGTools(data_root="/mnt/d/EEG", output_root="./agent_runs")
result = tools.invoke("extract_features", {
    "source": "/mnt/d/EEG/processed/sub01_preprocessed.npz",
    "max_windows": 128,
    "band_de_method": "filtered"
})
print(result["output_csv"])
print(result["features_npz"])
print(result["manifest_json"])
```

The source must be within `data_root` or `output_root`. `extract_features` requires **windowed EEG NPZ**, not EDF/BDF; first call `preprocess_eeg` for raw EEG. Outputs land in `output_root/features/<uuid>/`:

- `handcrafted_features.csv`: one row per channel × window, with `window_index`, subject/dataset/label (when supplied by source), window times, channel name and all numeric features.
- `handcrafted_features.npz`: numerical `features`, `feature_names`, `feature_units`, `channel_names`, `window_indices`, `sfreq`, plus available aligned subject ID / event label / dataset / time metadata. No pickle required.
- `feature_manifest.json`: definitions/units, total band, extraction method, output shape, aggregated feature statistics, and missing-value counts.

**The result uses `features_npz`, NOT `output_npz`**: other tools and the LangGraph state retain `latest_npz` as a windowed EEG signal file. This protects downstream graph composition.

## 3. Agent / LangGraph Tool Calling

The `extract_features` tool is registered automatically in `EEGTools.registry` and included in `tools.openai_schemas()` for ReAct. `Step.tool` also allows it in Plan-and-Execute and ReWOO.

The user can request:

> Preprocess the current EEG to 26 channels, 250 Hz, 1-second epochs, then extract all handcrafted features including Gaussian DE and relative delta/theta/alpha/beta/gamma band powers. Save the feature table.

An example tool plan (schema showing the relevant steps):

```json
{"steps":[
  {"id":"E1", "tool":"preprocess_eeg", "args":{"source":"$input", "target_count":26, "target_sfreq":250, "window_seconds":1.0}, "depends_on":[]},
  {"id":"E2", "tool":"extract_features", "args":{"source":"$E1.output_npz", "max_windows":128}, "depends_on":["E1"]}
]}
```

For existing processed NPZ, skip E1 and pass `$input` to `extract_features`. As with the other tools, the LLM only **chooses the tool and arguments**; numerical features are computed by SciPy and NumPy.

### Research cautions

- Verify measurement units and electrode references. Differential entropy depends on scale; referenced EEG also changes variance.
- Avoid treating overlapping windows from one subject as independent train/test subjects; group by subject for splits.
- Default features are computed **per channel per window**, not subject-level summaries; aggregate/group **after** subject-level train/test splitting.
- Short windows or nonstationary recording segments can distort Welch delta power, filtered DE, and Hjorth values. No feature guarantees clinical validity.
- `NaN` features occur for invalid/constant variance or unavailable power; inspect missing counts and choose a missing-data policy on **training data only** to prevent leakage.
