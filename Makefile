# AV-TalkerRFS -- reproduce everything.
#
#   make test   the test suite (seconds)
#   make all    every experiment plus the README tables (~20 min)
#   make clean  remove generated results and figures

PY  := python3
EXP := experiments

.PHONY: all test experiments tables clean

all: experiments tables

test:
	$(PY) -m pytest tests/ -q
	$(PY) scripts/smoke.py

experiments:
	cd $(EXP) && $(PY) exp1_montecarlo.py
	cd $(EXP) && $(PY) exp2_regimes.py
	cd $(EXP) && $(PY) exp3_complementarity.py

tables:
	$(PY) scripts/make_tables.py

clean:
	rm -f results/*.json results/*.log results/figures/*.png
	find . -name '__pycache__' -type d -exec rm -rf {} +
