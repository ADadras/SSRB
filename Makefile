.PHONY: test smoke lint

test:
	python -m pytest -q

smoke:
	python -m ssrb.cli smoke --output outputs/smoke-$$(date +%s)

lint:
	python -m ruff check src tests scripts
