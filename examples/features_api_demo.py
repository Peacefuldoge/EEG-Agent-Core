"""Run: python examples/features_api_demo.py (after installing Core)."""
from pathlib import Path
from eeg_agent import EEGTools, extract_handcrafted_features

root = Path(__file__).resolve().parent
output = root / 'feature_demo_outputs'
tools = EEGTools(data_root=root / 'demo_npz', output_root=output)
source = root / 'demo_npz' / 'synthetic_TDBRAIN.npz'

result = tools.invoke('extract_features', {
    'source': str(source), 'max_windows': 5,
    'band_de_method': 'filtered'
})
print('Matrix shape:', result['shape'])
print('Features:', result['feature_names'])
print('CSV:', result['output_csv'])
print('NPZ:', result['features_npz'])
print('Manifest:', result['manifest_json'])
