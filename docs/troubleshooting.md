# DataQA Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using DataQA.

## Common Issues

### Installation Issues

#### Issue: `pip install dataqa` fails

**Symptoms:**
- Package not found error
- Dependency conflicts
- Build failures

**Solutions:**

1. **Update pip and setuptools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Use virtual environment:**
   ```bash
   python -m venv dataqa-env
   source dataqa-env/bin/activate  # On Windows: dataqa-env\Scripts\activate
   pip install dataqa
   ```

3. **Install from source:**
   ```bash
   git clone https://github.com/your-org/dataqa.git
   cd dataqa
   pip install -e .
   ```

#### Issue: Import errors after installation

**Symptoms:**
```python
ImportError: No module named 'dataqa'
ModuleNotFoundError: No module named 'dataqa.agent'
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep dataqa
   python -c "import dataqa; print(dataqa.__version__)"
   ```

2. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   ```

3. **Reinstall in development mode:**
   ```bash
   pip uninstall dataqa
   pip install -e .
   ```

### Configuration Issues

#### Issue: Configuration validation errors

**Symptoms:**
```
ValidationError: 1 validation error for AgentConfig
llm.api_key
  field required
```

**Solutions:**

1. **Check required fields:**
   ```yaml
   # Ensure all required fields are present
   name: "my-agent"
   llm:
     provider: "openai"
     model: "gpt-3.5-turbo"
     api_key: "${OPENAI_API_KEY}"  # Required
   ```

2. **Verify environment variables:**
   ```bash
   echo $OPENAI_API_KEY
   # Should output your API key
   ```

3. **Use explicit values for testing:**
   ```yaml
   llm:
     api_key: "sk-your-actual-key-here"  # For testing only
   ```

#### Issue: Environment variable substitution not working

**Symptoms:**
- API key shows as literal `${OPENAI_API_KEY}`
- Authentication errors

**Solutions:**

1. **Set environment variables properly:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # Verify
   echo $OPENAI_API_KEY
   ```

2. **Use .env file:**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-key-here" > .env
   
   # Load in Python
   from dotenv import load_dotenv
   load_dotenv()
   ```

3. **Check variable syntax:**
   ```yaml
   # Correct
   api_key: "${OPENAI_API_KEY}"
   
   # Incorrect
   api_key: "$OPENAI_API_KEY"
   api_key: "{OPENAI_API_KEY}"
   ```

### LLM Issues

#### Issue: OpenAI API authentication errors

**Symptoms:**
```
AuthenticationError: Incorrect API key provided
RateLimitError: Rate limit exceeded
```

**Solutions:**

1. **Verify API key:**
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

2. **Check API key format:**
   - Should start with `sk-`
   - Should be 51 characters long
   - No extra spaces or characters

3. **Handle rate limits:**
   ```yaml
   llm:
     max_retries: 5
     timeout: 60.0
     extra_params:
       request_timeout: 30
   ```

#### Issue: LLM responses are inconsistent or poor quality

**Symptoms:**
- Generated code doesn't work
- Responses don't match the query
- Inconsistent behavior

**Solutions:**

1. **Adjust temperature:**
   ```yaml
   llm:
     temperature: 0.1  # Lower for more consistent responses
   ```

2. **Increase context:**
   ```yaml
   workflow:
     max_context_length: 8000  # More context for better responses
   ```

3. **Add more knowledge:**
   ```python
   # Add schema and business context
   docs = [
       Document(
           content="Detailed table schema and relationships",
           metadata={"type": "schema"},
           source="schema.md"
       )
   ]
   agent.ingest_knowledge(docs)
   ```

### Execution Issues

#### Issue: Code execution timeouts

**Symptoms:**
```
ExecutionError: Code execution timed out after 30.0 seconds
```

**Solutions:**

1. **Increase timeout:**
   ```yaml
   executor:
     max_execution_time: 120.0  # 2 minutes
   ```

2. **Optimize queries:**
   - Add database indexes
   - Limit result sets
   - Use more efficient queries

3. **Check resource usage:**
   ```yaml
   executor:
     max_memory_mb: 2048  # Increase memory limit
     max_rows: 100000     # Increase row limit
   ```

#### Issue: Memory errors during execution

**Symptoms:**
```
MemoryError: Unable to allocate memory
ExecutionError: Memory limit exceeded
```

**Solutions:**

1. **Increase memory limits:**
   ```yaml
   executor:
     max_memory_mb: 4096  # 4GB
   ```

2. **Process data in chunks:**
   ```python
   # In generated code, process data in smaller chunks
   chunk_size = 10000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       # Process chunk
   ```

3. **Use more efficient data types:**
   ```python
   # Optimize pandas dtypes
   df = df.astype({
       'id': 'int32',
       'category': 'category',
       'amount': 'float32'
   })
   ```

#### Issue: Database connection errors

**Symptoms:**
```
ConnectionError: Unable to connect to database
OperationalError: database is locked
```

**Solutions:**

1. **Check database configuration:**
   ```yaml
   executor:
     database_url: "sqlite:///path/to/database.db"
     # Or for other databases
     database_url: "postgresql://user:pass@host:port/db"
   ```

2. **Verify database permissions:**
   ```bash
   # Check file permissions
   ls -la database.db
   
   # Test connection
   sqlite3 database.db ".tables"
   ```

3. **Handle connection pooling:**
   ```yaml
   executor:
     connection_pool_size: 5
     connection_timeout: 30.0
   ```

### Knowledge Base Issues

#### Issue: Knowledge ingestion fails

**Symptoms:**
```
KnowledgeError: Failed to ingest documents
EmbeddingError: Unable to generate embeddings
```

**Solutions:**

1. **Check document format:**
   ```python
   # Ensure documents are properly formatted
   doc = Document(
       content="Valid text content",  # Required
       metadata={"type": "schema"},   # Optional but recommended
       source="source.md"             # Required
   )
   ```

2. **Verify embedding model:**
   ```yaml
   knowledge:
     embedding_model: "all-MiniLM-L6-v2"  # Ensure model exists
   ```

3. **Check document size:**
   ```python
   # Split large documents
   if len(document.content) > 1000:
       # Split into smaller chunks
       chunks = split_document(document, chunk_size=512)
   ```

#### Issue: Poor knowledge retrieval

**Symptoms:**
- Relevant context not found
- Irrelevant results returned
- Knowledge not used in responses

**Solutions:**

1. **Adjust similarity threshold:**
   ```yaml
   knowledge:
     similarity_threshold: 0.6  # Lower for more results
     top_k: 10                  # More results
   ```

2. **Improve document quality:**
   ```python
   # Add better metadata
   doc = Document(
       content="Clear, specific content",
       metadata={
           "type": "schema",
           "table": "customers",
           "importance": "high"
       },
       source="schema.md"
   )
   ```

3. **Use better embedding model:**
   ```yaml
   knowledge:
     embedding_model: "all-mpnet-base-v2"  # Higher quality
   ```

### Workflow Issues

#### Issue: Approval workflow not working

**Symptoms:**
- Operations execute without approval
- Approval prompts not shown
- Cannot approve/reject operations

**Solutions:**

1. **Check workflow configuration:**
   ```yaml
   workflow:
     require_approval: true
     auto_approve_safe: false  # Ensure manual approval
   ```

2. **Implement approval handler:**
   ```python
   # Check for pending approvals
   if agent.has_pending_approval():
       operation = agent.get_pending_operation()
       print(f"Approve operation: {operation}")
       
       # Get user input
       approval = input("Approve? (y/n): ").lower() == 'y'
       agent.approve_operation(approved=approval)
   ```

3. **Debug approval state:**
   ```python
   # Check agent state
   status = agent.get_conversation_status()
   print(f"Pending approval: {status.get('pending_approval')}")
   ```

## Debugging Techniques

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configuration
log_level: "DEBUG"
```

### Use Debug Mode

```yaml
workflow:
  debug_mode: true  # Enables detailed logging
```

### Inspect Agent State

```python
# Get detailed agent information
info = agent.get_agent_info()
print(f"Agent info: {info}")

# Check conversation history
history = agent.get_conversation_history()
print(f"Conversation: {history}")

# Inspect generated code
if hasattr(agent, 'last_generated_code'):
    print(f"Last code: {agent.last_generated_code}")
```

### Test Components Individually

```python
# Test LLM directly
from dataqa.primitives import LLMInterface
llm = LLMInterface(config.llm)
response = llm.generate("Test prompt")

# Test executor directly
from dataqa.primitives import InMemoryExecutor
executor = InMemoryExecutor(config.executor)
result = executor.execute_sql("SELECT 1")

# Test knowledge base
from dataqa.primitives import FAISSKnowledge
knowledge = FAISSKnowledge(config.knowledge)
results = knowledge.search("test query")
```

## Performance Issues

### Slow Query Processing

**Symptoms:**
- Long response times
- High CPU/memory usage
- Timeouts

**Solutions:**

1. **Optimize LLM settings:**
   ```yaml
   llm:
     model: "gpt-3.5-turbo"  # Faster than GPT-4
     max_tokens: 1000        # Limit response length
     timeout: 30.0           # Reasonable timeout
   ```

2. **Reduce context size:**
   ```yaml
   workflow:
     max_context_length: 4000  # Smaller context
   knowledge:
     top_k: 5                  # Fewer knowledge results
   ```

3. **Use caching:**
   ```yaml
   cache_dir: "./cache"  # Enable caching
   ```

### High Memory Usage

**Solutions:**

1. **Limit data processing:**
   ```yaml
   executor:
     max_rows: 10000      # Limit result size
     max_memory_mb: 1024  # Memory limit
   ```

2. **Use streaming for large datasets:**
   ```python
   # Process data in chunks
   for chunk in pd.read_csv('large_file.csv', chunksize=1000):
       process_chunk(chunk)
   ```

## Getting Help

### Check Logs

```bash
# Application logs
tail -f dataqa.log

# System logs
journalctl -u dataqa-service
```

### Collect Debug Information

```python
# Generate debug report
debug_info = {
    'version': dataqa.__version__,
    'config': agent.config.dict(),
    'system_info': platform.platform(),
    'python_version': sys.version,
    'dependencies': [str(req) for req in pkg_resources.working_set]
}
print(json.dumps(debug_info, indent=2))
```

### Community Support

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-org/dataqa/issues)
- **Discord**: [Join our community](https://discord.gg/dataqa)
- **Documentation**: [Read the full docs](https://docs.dataqa.dev)
- **Stack Overflow**: Tag questions with `dataqa`

### Professional Support

For enterprise customers:
- **Email**: support@dataqa.dev
- **Slack**: Enterprise Slack channel
- **Phone**: Available for enterprise plans

## Frequently Asked Questions

### Q: Can I use DataQA without an internet connection?

A: Partially. You can use local LLM models and local knowledge bases, but some features require internet connectivity.

### Q: How do I secure my API keys?

A: Use environment variables, never commit keys to version control, and consider using secret management services in production.

### Q: Can I use multiple LLM providers?

A: Yes, you can configure different agents with different LLM providers and switch between them.

### Q: How do I handle large datasets?

A: Use chunking, streaming, and appropriate memory limits. Consider using API-based executors for very large datasets.

### Q: Is DataQA suitable for production use?

A: Yes, with proper configuration, security measures, and monitoring. Use the enterprise configuration template as a starting point.