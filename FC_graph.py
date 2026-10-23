"""Legacy-friendly FC circle entrypoint. Usage: python FC_graph.py --input-file x.npz --output-dir figures"""
from eeg_visualization import main

if __name__ == '__main__':
    main(preset_plots='fc_circle')
