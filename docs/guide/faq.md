# Frequently Asked Questions (FAQ)

Find answers to common questions about using, configuring, and extending DataQA.

---

## General

**Q: What is DataQA?**
A: DataQA is a modular Python framework for building intelligent data agents and pipelines that can answer natural language questions about your data.

**Q: Who should use DataQA?**
A: Data scientists, ML engineers, and developers who want to build conversational analytics, data chatbots, or custom data pipelines.

---

## Installation & Setup

**Q: What Python version is required?**
A: Python 3.11 or higher.

**Q: How do I install DataQA?**
```bash
pip install aicoelin-dataqa
```
See [Installation Guide](../installation.md) for details.

**Q: How do I set LLM credentials?**
A: Set the required environment variables (e.g., `AZURE_OPENAI_API_KEY`, `OPENAI_API_BASE`) in your shell or `.env` file.
See [Installation Guide#set-up-environment-variables](../installation.md#set-up-environment-variables).

---

## Configuration

**Q: Where do I put my schema, rules, and example files?**
A: In the directory specified by `asset_directory` in your agent config (commonly `data/`).

**Q: Can I use environment variables in my YAML config?**
A: Yes! Use `${VAR}` or `${VAR:-default}` syntax.

**Q: How do I use a different LLM for planning vs. execution?**
A: Assign different LLMs in the `llm` section of your agent config.
See [Customizing Agents](customizing_agents.md).

---

## Running & Debugging

**Q: How do I run an example agent?**
```bash
python -m dataqa.examples.cib_mp.agent.cwd_agent
```
See [Running the Examples](running_examples.md).

**Q: My agent returns empty or irrelevant results. What should I do?**
A:
- Add more rules or in-context examples to guide the LLM.
- Check your schema and data files for completeness.
- Revise your agent config for typos.

**Q: How do I debug YAML or config errors?**

## Extending & Customization

**Q: Can I add my own analytics or plot tools?**
A: Yes! Register your tool in the tool registry.
See [Extending DataQA](extending.md).

**Q: How do I use a custom LLM or SQL executor?**
A: Subclass the appropriate base class and reference it in your config.
See [Extending DataQA](extending.md).

---

## Deployment

**Q: How can I run DataQA in Docker or the cloud?**
A: Yes! See [Deployment Guide](deployment.md) for Docker, Kubernetes, and cloud best practices.

---

## Support

**Q: Where can I get help?**
A:
- [Troubleshooting Guide](troubleshooting.md)
- [TODO] - Add a support mailing list.

---

## Contributing

**Q: How can I contribute to DataQA?**
A: See [Contributing Guide](../contributing.md) for setup, code style, and PR process.
