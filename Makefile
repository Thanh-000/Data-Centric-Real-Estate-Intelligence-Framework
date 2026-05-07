PYTHON ?= python

.PHONY: install quickstart download run health analyze-abstention evaluate-slices dashboard notebook test clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

quickstart:
	$(PYTHON) scripts/quickstart.py --install

download:
	$(PYTHON) scripts/download_data.py

run:
	$(PYTHON) scripts/run_pipeline.py

health:
	$(PYTHON) scripts/health_check.py

analyze-abstention:
	$(PYTHON) scripts/analyze_abstention.py

evaluate-slices:
	$(PYTHON) scripts/evaluate_slices.py

dashboard:
	streamlit run app/streamlit_app.py

notebook:
	$(PYTHON) -m notebook notebooks/01_dc_reif_king_county.ipynb

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('data/interim'), Path('data/processed'), Path('data/artifacts'), Path('outputs')]]"
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/interim','data/processed','data/artifacts','outputs/figures','outputs/tables','outputs/reports']]"
