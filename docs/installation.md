# Installation

This guide will help you install DataQA and set up your environment to run the CWD Agent.

---

## Prerequisites

- **Python:** 3.11 or higher
- **Package Manager:** pip

---

# 1. Install DataQA

Install the latest version of the library from PyPI using pip. It's recommended to do this in a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install the library
pip install aicoelin-dataqa
```

---

# 2. Set Up Environment Variables

The DataQA Agent needs credentials to access a Large Language Model (LLM), such as Azure OpenAI. You must set these variables before running your agent.

### **Method 1: In your shell (Recommended for testing)**

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"

# Optional: For certificate-based authentication
export CLIENT_ID="your-azure-client-id"
export TENANT_ID="your-azure-tenant-id"
export CERT_PATH="/path/to/your/certificate.pem"
```

### **Method 2: In a `.env` file (Recommended for projects)**

Create a file named `.env` in your project's root directory:

```
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"

# Optional cert auth variables
CLIENT_ID="your-azure-client-id"
TENANT_ID="your-azure-tenant-id"
CERT_PATH="/path/to/your/certificate.pem"
```
The library will automatically load these variables if `python-dotenv` is installed.

---

# 3. Verify Your Installation

You can quickly check that the library is installed and accessible:

```bash
python -c "import dataqa; print(f'DataQA version: {dataqa.__version__}')"
```

If this command prints the version number without errors, your installation was successful.

---

# Next Steps

You are now ready to build and run your first agent!

- **[Quickstart Guide](quickstart.md)**: Your next stop to get an agent running immediately.
