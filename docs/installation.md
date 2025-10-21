# Installation

This page will help you install DataQA and set up your environment for local development or production use.

---

## Supported Platforms

- **Python:** 3.11 or higher
- **Package Managers:** pip, Poetry

---

# 1. Install DataQA

### **Using pip (Recommended)**

```bash
pip install aicoelin-dataqa
```

### **Using Poetry**

```bash
poetry add aicoelin-dataqa
```

### **From Source (Development)**

```bash
git clone https://bitbucketdc.jpmchase.net/scm/aicoelin/dataqa-lib.git
cd dataqa-lib
poetry install
```

# 2. Set Up Environment Variables

DataQA requires environment variables for LLM access (e.g., Azure OpenAI).
**Set these before running your agent or pipeline.**

### **In your shell:**

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
# Optional, if using Azure AD or certificate-based auth:
export AZURE_OPENAI_API_TOKEN="your-azure-ad-auth-token"
export CLIENT_ID="your-azure-tenant-id"
export CERT_PATH="/path/to/your/cert.pem"
```


### **Or in a `.env` file:**

```
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
AZURE_OPENAI_API_TOKEN="your-azure-ad-auth-token"
CLIENT_ID="your-azure-tenant-id"
TENANT_ID="your-azure-tenant-id"
CERT_PATH="/path/to/your/cert.pem"
```

> **Tip:** Use [python-dotenv](https://pypi.org/project/python-dotenv/) or your IDE to load `.env` files automatically.

---

# 3. Verify Your Installation

Check that DataQA is installed and your environment is set up:

```bash
python -c "import dataqa; print(dataqa.__version__)"
```

If you see the version number, you're ready to go!

---

# 4. Run an Example to Verify Everything Works

After installation, you can quickly verify your setup by running one of the included DataQA examples.

### **Run the CIB Merchant Payments Example**

#### **1. Set your environment variables** (see above).

#### **2. Run the example script:**

```bash
python -m dataqa.examples.cib_mp.agent.cwd_agent
```

This will:
- Initialize a sample agent using the configuration in `dataqa/examples/cib_mp/agent/`
- Run a pre-defined query
- Print the agent's response and execution trace to your console

#### **3. Expected Output:**
- You should see a final text response, any output dataframes, and a debug trace.
- If you see authentication errors, check your environment variables.

> **Tip:** You can open and modify the script at `dataqa/examples/cib_mp/agent/cwd_agent.py` to try your own queries.

---

# 5. (Optional) Install Development Tools

For contributors or advanced users:

```bash
# Install Poetry if you don't have it
pip install poetry

# Install pre-commit hooks, linters, and formatters
make ci-prebuild
make lint-format
make precommit
```

---

# 6. Troubleshooting

---

## **Missing LLM credentials**
If you see authentication errors, double-check your environment variables.

## **Unsupported Python version**
DataQA requires Python 3.11+. Check with `python --version`.

## **Dependency issues**
Try upgrading pip: `pip install --upgrade pip`.

---

# Next Steps

- [Quickstart Guide](quickstart.md)
- [User Guide](guide/introduction.md)
- [Building Your First Agent](guide/building_your_first_agent.md)

---

# Need Help?

- [FAQ](guide/faq.md)
