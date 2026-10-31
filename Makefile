.PHONY: test audit lint

test:
	.venv/bin/pytest

audit:
	.venv/bin/python scripts/offline_audit.py

lint:
	.venv/bin/ruff check .

