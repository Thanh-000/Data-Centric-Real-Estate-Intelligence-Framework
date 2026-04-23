PYTHON ?= python

.PHONY: install download run notebook test clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_data.py

run:
	$(PYTHON) scripts/run_pipeline.py

notebook:
	$(PYTHON) -m notebook notebooks/01_dc_reif_king_county.ipynb

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('data/interim'), Path('data/processed'), Path('data/artifacts'), Path('outputs')]]"
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/interim','data/processed','data/artifacts','outputs/figures','outputs/tables','outputs/reports']]"

