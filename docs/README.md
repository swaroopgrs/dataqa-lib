# Running Documentation Locally

To preview and test the documentation locally, follow these steps:

## Prerequisites

Make sure you have the project dependencies installed. If using Poetry:

```bash
poetry install
```

Or if using pip:

```bash
pip install -e ".[dev]"
```

This will install mkdocs and all required documentation dependencies.

## Running the Documentation Server

### Using Poetry

```bash
poetry run mkdocs serve
```

### Using pip/virtual environment

```bash
mkdocs serve
```

## Accessing the Documentation

Once the server starts, you'll see output like:

```
INFO     -  Building documentation...
INFO     -  Cleaning site directory
INFO     -  Documentation built in X.XX seconds
INFO     -  [XX:XX:XX] Serving on http://127.0.0.1:8000
```

Open your browser and navigate to `http://127.0.0.1:8000` to view the documentation.

## Live Reload

The `mkdocs serve` command automatically watches for file changes and reloads the documentation. Just save your markdown files and refresh your browser to see the changes.

## Building Static Site

To build a static version of the documentation (for deployment):

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML files.

## Troubleshooting

- **Module not found errors**: Make sure you've installed all dependencies with `poetry install` or `pip install -e ".[dev]"`
- **Port already in use**: Use `mkdocs serve -a 127.0.0.1:8001` to use a different port
- **YAML errors**: Check `mkdocs.yml` for syntax errors


