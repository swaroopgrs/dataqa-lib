POETRY_VERSION=2.1.1
SSAP_DIR = ssap_bill_of_materials
SSAP_BOM = $(SSAP_DIR)/pip_dependency_tree.txt

.PHONY: clean build dist test precommit ssap ci-prebuild package ci lint-format check-lint-format help
.DEFAULT_GOAL=help
.NOTPARALLEL: ; # Targets execute serially
.ONESHELL: ; # Recipes execute in the same shell

clean: ## Remove all cache, reports, coverage, distrubution files and folders
	rm -f .coverage
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf reports
	rm -f requirements.txt
	rm -rf $(SSAP_DIR)

env-clean:
	clean
	rm -rf .venv

build: ## Create the virtual environment and install development dependencies
	python -m poetry install

dist: ## Build source and built distribution package
	python -m poetry build

pre-test: # Install dependencies for testing
	python -m pip install pytest pytest-asyncio

test: ## Executes test cases and produce coverage reports
	python -m poetry run pytest --cov . --junitxml reports/xunit.xml --cov-report xml:reports/coverage.xml \
		--cov-report html:reports/htmlcov --cov-report term-missing
	
precommit: ## Installs and runs pre-commit
	python -m poetry run pre-commit install 
	python -m poetry run pre-commit run --all-files

ssap: ## Generate requirements.txt file and required reports for SSAP
	mkdir -p $(SSAP_DIR)
	pip install poetry-plugin-export
	python -m poetry export --without-hashes -o $(SSAP_BOM)

ci-prebuild: ## Install build tools and prepare project directory for the CI pipeline
	python -m pip install -U pip setuptools poetry==$(POETRY_VERSION) poetry-plugin-export
	cat /dev/null > requirements.txt

package: python -m poetry build --format wheel ## Create deployable whl packages for python project

ci: clean build package ## Runs clean, build and pacakge

lint-format: ## Formats and Lints the Python files
	python -m poetry run ruff format .
	python -m poetry run ruff check .

check-lint-format: ## Check format for all files
	python -m poetry run ruff check --force-exclude
	python -m poetry run ruff format --check --exclude "**/*.ipynb"
	
help: ## Show make target documentation
	@awk -F ':|##' '/^[^\t].+?:.*?##/ {\
	printf "\033[36m%-30s\033[0m %s\n", $$1, $$NF \
	}' ${MAKEFILE_LIST}