---
inclusion: always
---

# DataQA Testing & Development Commands

## Core Testing Commands

### Running Tests
```bash
# Run all tests with coverage
uv run pytest

# Run tests without coverage
uv run pytest --no-cov

# Run specific test file
uv run pytest tests/models/test_execution.py

# Run specific test function
uv run pytest tests/models/test_execution.py::test_execution_creation

# Run tests with specific markers
uv run pytest -m unit          # Only unit tests
uv run pytest -m integration   # Only integration tests
uv run pytest -m "not slow"    # Skip slow tests

# Run tests in parallel (if pytest-xdist is installed)
uv run pytest -n auto

# Run tests with verbose output
uv run pytest -v

# Run tests and stop on first failure
uv run pytest -x
```

### Coverage Commands
```bash
# Generate HTML coverage report (opens in htmlcov/index.html)
uv run pytest --cov-report=html

# Generate terminal coverage report
uv run pytest --cov-report=term-missing

# Generate XML coverage report (for CI/CD)
uv run pytest --cov-report=xml

# Coverage for specific module
uv run pytest --cov=src/dataqa/models
```

## Code Quality Commands

### Linting & Formatting
```bash
# Check code style with Ruff
uv run ruff check

# Fix auto-fixable issues
uv run ruff check --fix

# Format code with Ruff
uv run ruff format

# Check specific files
uv run ruff check src/dataqa/models/
uv run ruff format tests/
```

### Type Checking
```bash
# Run MyPy type checking
uv run mypy src/dataqa

# Type check specific module
uv run mypy src/dataqa/models/

# Type check with verbose output
uv run mypy --verbose src/dataqa
```

## Development Workflow Commands

### Environment Management
```bash
# Sync dependencies (install/update)
uv sync

# Install development dependencies
uv sync --group dev

# Add new dependency
uv add pandas>=2.2.0

# Add development dependency
uv add --group dev pytest-mock

# Remove dependency
uv remove package-name

# Show dependency tree
uv tree
```

### LangGraph Development
```bash
# Start LangGraph development server
uv run langgraph dev

# Run LangGraph with specific config
uv run langgraph dev --config config/example_agent.yaml

# LangGraph CLI help
uv run langgraph --help
```

### DataQA CLI Commands
```bash
# Run DataQA CLI
uv run python -m dataqa.cli --help

# Alternative CLI access
uv run dataqa --help
```

## Continuous Integration Commands

### Pre-commit Hooks
```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff
uv run pre-commit run mypy
```

### Full Quality Check Pipeline
```bash
# Complete quality check (run before committing)
uv run ruff check --fix && \
uv run ruff format && \
uv run mypy src/dataqa && \
uv run pytest
```

## Debugging & Development

### Interactive Testing
```bash
# Run tests with Python debugger
uv run pytest --pdb

# Run tests with IPython debugger (if installed)
uv run pytest --pdbcls=IPython.terminal.debugger:Pdb

# Run single test with debugging
uv run pytest -s tests/models/test_execution.py::test_specific_function
```

### Performance Testing
```bash
# Run tests with timing information
uv run pytest --durations=10

# Profile test execution
uv run pytest --profile

# Run only fast tests (exclude slow marker)
uv run pytest -m "not slow"
```

## Environment-Specific Commands

### Local Development
```bash
# Activate virtual environment manually
source .venv/bin/activate

# Run commands in activated environment
pytest
ruff check
mypy src/dataqa
```

### Docker/Container Testing
```bash
# If using Docker for testing
docker run --rm -v $(pwd):/app -w /app python:3.11 \
  bash -c "pip install uv && uv sync && uv run pytest"
```

## Useful Testing Patterns

### Watch Mode (requires pytest-watch)
```bash
# Auto-run tests on file changes
uv add --group dev pytest-watch
uv run ptw
```

### Test Data Management
```bash
# Clean test artifacts
rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml

# Reset test database/cache
rm -rf cache/ data/test_*
```

## Quick Reference

**Most Common Commands:**
- `uv run pytest` - Run all tests
- `uv run ruff check --fix` - Fix code style
- `uv run mypy src/dataqa` - Type check
- `uv sync` - Update dependencies
- `uv run langgraph dev` - Start development server

**Before Committing:**
```bash
uv run ruff check --fix && uv run ruff format && uv run mypy src/dataqa && uv run pytest
```