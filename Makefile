PYTHON ?= python

.PHONY: install download run report-results diagnostics notebook test smoke clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_data.py

run:
	$(PYTHON) scripts/run_pipeline.py

report-results:
	$(PYTHON) scripts/build_report_results.py

diagnostics:
	$(PYTHON) scripts/build_diagnostics.py

notebook:
	$(PYTHON) -m notebook notebooks/01_dc_reif_king_county.ipynb

test:
	$(PYTHON) -m pytest -q

smoke:
	bash scripts/smoke_test.sh

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('data/interim'), Path('data/processed'), Path('data/artifacts'), Path('outputs')]]"
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/interim','data/processed','data/artifacts','outputs/figures','outputs/tables','outputs/reports','outputs/final_report_pack']]"
