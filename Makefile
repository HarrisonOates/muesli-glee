.PHONY: test audit lint paper clean

test:
	.venv/bin/pytest

audit:
	.venv/bin/python scripts/offline_audit.py

lint:
	.venv/bin/ruff check .

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	cd paper && latexmk -C main.tex

