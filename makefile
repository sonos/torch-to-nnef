sources = torch_to_nnef packages/llm/torch_to_nnef_llm packages/nemo-asr/torch_to_nnef_nemo

.PHONY: test format lint prose unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources) tests
	ruff format $(sources) tests

# `prose` is a prerequisite rather than a recipe line: as a trailing line it
# was unreachable, because the mypy step above it currently exits non-zero.
lint: prose
	ruff check $(sources) tests
	mypy $(sources) tests

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
