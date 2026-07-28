sources = torch_to_nnef packages/llm/torch_to_nnef_llm packages/nemo-asr/torch_to_nnef_nemo

.PHONY: test format lint prose unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources) tests
	ruff format $(sources) tests

lint:
	ruff check $(sources) tests
# Before mypy, which currently exits non-zero (so a trailing line here would be
# unreachable), and after ruff, so a stray em dash cannot hide code findings.
	python3 .github/scripts/check_prose.py
# Not `tests`: CONTRIBUTING documents mypy as type-checking library code
# "excluding tests", and [tool.mypy] excludes it, so passing it here made
# mypy fail with "no .py files in directory".
	mypy $(sources)

prose:
	python3 .github/scripts/check_prose.py

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
