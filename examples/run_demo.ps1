# Run in repository root with Python dependencies installed.
python examples/create_demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python eeg_visualization.py --input-dir examples/demo_npz --output-dir examples/demo_report `
  --plots all --max-windows 12 --compare --dpi 150
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Open examples/demo_report/index.html"
