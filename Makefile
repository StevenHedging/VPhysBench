.PHONY: test smoke

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python -m physbench smoke --output-root runs/smoke

