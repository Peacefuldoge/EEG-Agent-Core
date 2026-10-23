"""Legacy-friendly FC matrix entrypoint. Usage: python FC_Matrix.py --input-file x.npz --output-dir figures"""
from eeg_visualization import main

if __name__ == '__main__':
    main(preset_plots='fc_heatmap')
