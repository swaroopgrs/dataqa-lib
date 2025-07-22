#!/bin/bash
# Setup script for DataQA with uv dependency management

set -e

echo "🚀 Setting up DataQA with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ uv found: $(uv --version)"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    echo "   Please install Python 3.11+ or use uv to manage Python versions:"
    echo "   uv python install 3.11"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment with uv
echo "📦 Creating virtual environment..."
uv venv --python 3.11

# Install dependencies
echo "📥 Installing dependencies..."
uv sync

# Install development dependencies
echo "🛠️  Installing development dependencies..."
uv sync --group dev

# Install LangGraph CLI
echo "🔧 Installing LangGraph CLI..."
uv add --group dev "langgraph-cli[inmem]"

echo ""
echo "✅ Setup complete! 🎉"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use uv to run commands directly:"
echo "  uv run python -m dataqa.cli --help"
echo "  uv run pytest"
echo "  uv run langgraph dev"
echo ""
echo "To add new dependencies:"
echo "  uv add <package-name>"
echo "  uv add --group dev <dev-package-name>"