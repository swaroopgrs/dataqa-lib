# Troubleshooting

Having trouble with DataQA?
This guide covers common issues and how to resolve them.

---

## 1. Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'dataqa'`
**Solution:**
- Make sure you have installed DataQA:
```bash
pip install aicoelin-dataqa
```
- If using a virtual environment, ensure it's activated.

**Problem:** `ERROR: Could not find a version that satisfies the requirement ...`
**Solution:**
- DataQA requires Python 3.11 or higher.
  Check your version with `python --version`.

---

## 2. Environment Variable & Credential Issues

**Problem:** `Authentication Failed` or `401 Unauthorized`
**Solution:**
- Double-check your environment variables for LLM credentials:
  - `AZURE_OPENAI_API_KEY`
  - `OPENAI_API_BASE`
  - (Others as needed)
- If using a `.env` file, ensure it's loaded (try `print(os.environ)` in Python).

**Problem:** `KeyError: 'AZURE_OPENAI_API_KEY'`
**Solution:**
- Set the required environment variable in your shell or `.env` file.

---

## 3. Configuration & YAML Issues

**Problem:** `FileNotFoundError .../schema.yml`
**Solution:**
- Ensure all referenced files exist and paths are correct in your config.

**Problem:** `yaml.scanner.ScannerError` or `YAMLError`
**Solution:**
- Check your YAML for syntax errors (indentation, colons, etc.).
- Use a YAML linter or IDE with YAML support.

---

## 4. Running Examples

**Problem:** Example script fails with authentication or missing file errors
**Solution:**
- Set all required environment variables.
- Ensure you have the correct data files in the expected locations.

---

## 5. Agent & Pipeline Errors

**Problem:** `TypeError`, `ValueError`, or unexpected output
**Solution:**
- Check your agent and asset YAML files for typos or missing fields.
- Review the [Configuration Guide](configuration.md) for required sections.

**Problem:** Agent returns empty or irrelevant results
**Solution:**
- Add more rules or in-context examples to guide the LLM.
- Check that your schema and data files are correct and complete.

---

## 6. LLM/Network Issues

**Problem:** `TimeoutError` or slow responses
**Solution:**
- Check your network connection.
- Try a smaller query or dataset.
- If using a cloud LLM, check for service outages.

---

## 7. Windows-Specific Issues

**Problem:** Path or encoding errors
**Solution:**
- Use WSL (Windows Subsystem for Linux) for best compatibility.
- Ensure file paths use forward slashes (`/`).

---

## 8. Still Stuck?

- Check the [FAQ](faq.md).

---

## Next Steps

- [FAQ](faq.md)
- [User Guide](introduction.md)
- [API Reference](../reference/agent.md)
