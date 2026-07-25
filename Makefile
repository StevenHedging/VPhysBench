.PHONY: test smoke

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python3 -m physbench smoke --output-root runs/smoke
