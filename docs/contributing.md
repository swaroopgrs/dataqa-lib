# Contributing to DataQA

Thank you for your interest in contributing to DataQA!
We welcome bug reports, feature requests, documentation improvements, and code contributions.

---

# 1. Development Setup

## **Prerequisites**

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/) for dependency management

## **Clone the Repository**

```bash
git clone https://bitbucketdc.jpmchase.net/scm/aicoelin/dataqa-lib.git
cd dataqa-lib
```

## **Install Dependencies**

```bash
poetry install
```

## **Set Up Pre-commit Hooks (Recommended)**

```bash
make ci-prebuild
make precommit
```

---

# 2. Code Style & Linting

- Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) and use type hints.
- Format code with [Black](https://black.readthedocs.io/en/stable/).
- Lint with [Flake8](https://flake8.pyqa.org/en/latest/) and [isort](https://pycqa.github.io/isort/).
- Run all checks before committing:

```bash
make lint-format
```

---

# 3. Running Tests

- All new code should include unit or integration tests.
- Run the test suite with:

```bash
pytest
```

- Add new tests in the `tests/` directory.

---

# 4. Documentation

- All public classes and functions should have docstrings.
- Documentation is in the `docs/` folder (Markdown).
- To build the docs locally (if using MkDocs):

```bash
mkdocs serve
```

---

# 5. Submitting a Pull Request

1. Fork the repository and create a new branch.
2. Make your changes with clear, atomic commits.
3. Add or update tests and documentation as needed.
4. Ensure all tests and linters pass.
5. Open a pull request with a clear description of your changes.

---

# 6. Reporting Bugs & Requesting Features

- Send an email to [TODO] for bug reports and feature requests.
- Please include as much detail as possible (error messages, config snippets, etc.).

---

# 7. Code of Conduct

Be respectful and inclusive.

---

# 8. Release Process (Maintainers)

- Update the `changelog.md` with new features and fixes.
- Bump the version in `pyproject.toml`.
- Tag the release and push to PyPI.

---

# 9. Need Help?

- [FAQ](/guide/faq.md)

---

Thank you for helping make DataQA better!