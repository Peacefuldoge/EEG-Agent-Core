"""Plugin-style EEG analysis without importing Chainlit, LangGraph or a model."""
from pathlib import Path
from eeg_agent import EEGTools

DATA=(Path(__file__).parent/'demo_npz').resolve()
OUT=(Path(__file__).parent.parent/'agent_runs').resolve()
tools=EEGTools(DATA,OUT)
source=DATA/'synthetic_TDBRAIN.npz'
print('QC:',tools.invoke('assess_eeg_quality',{'source':str(source)})['status'])
print('Integrity:',tools.invoke('validate_eeg',{'source':str(source)})['status'])
print('Proposal:',tools.invoke('propose_preprocessing',{'source':str(source)})['proposal'])
for method in ('pearson','coh','imcoh','plv','pli','wpli'):
    result=tools.invoke('analyze_connectivity',{
        'source':str(source),'method':method,'max_windows':16,
        'bands':{'alpha':(8,13)}
    })
    print(method,result['matrix_npz'],result['metrics_by_band'])

# Batch preprocessing: supply a directory containing CONTINUOUS recordings.
# tools.invoke('batch_preprocess', {
#     'source_dir':'/mnt/d/EEG/continuous',
#     'preprocessing':{'target_sfreq':250,'window_seconds':1,'use_ica':False},
#     'continue_on_error':True
# })
