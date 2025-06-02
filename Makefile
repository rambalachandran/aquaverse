.PHONY: all clean format lint test tests test_watch integration_tests docker_tests help

all: help

# PYTHON_FILES="model tests"
# lint: PYTHON_FILES="model tests"
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$')

format:
	ruff format --exclude '**/*.ipynb' index rag experiments

typecheck:
	mypy src test

lint:
	ruff check --fix --exclude '**/*.ipynb' index rag experiments

help:
	@echo '----'
	@echo 'format              - run code formatters with ruff'
	@echo 'lint                - run linters with ruff'
	@echo 'typcheck                - run type hint checkers with mypy'
