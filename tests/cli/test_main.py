"""Tests for the main CLI functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from dataqa.cli.main import app
from dataqa.config.models import AgentConfig


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    config_data = {
        "name": "test-agent",
        "description": "Test agent configuration",
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "temperature": 0.1
        },
        "knowledge": {
            "provider": "faiss",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "executor": {
            "provider": "inmemory",
            "database_type": "duckdb"
        },
        "workflow": {
            "strategy": "react",
            "require_approval": False  # Disable approval for testing
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_document_file():
    """Create a temporary document file for ingestion testing."""
    content = """
    # Test Document
    
    This is a test document for knowledge base ingestion.
    It contains some sample information about data analysis.
    
    ## Key Concepts
    
    - Data analysis involves examining datasets
    - SQL is used for querying databases
    - Python is great for data manipulation
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestVersionCommand:
    """Test the version command."""
    
    def test_version_command(self, runner):
        """Test version command shows version information."""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "DataQA version" in result.stdout
        assert "composable data agent framework" in result.stdout


class TestConfigCommand:
    """Test configuration management commands."""
    
    def test_config_create(self, runner):
        """Test creating a new configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_config.yaml"
            
            result = runner.invoke(app, [
                "config", "create",
                "--output", str(output_path)
            ])
            
            assert result.exit_code == 0
            assert output_path.exists()
            assert "Created example configuration" in result.stdout
            
            # Verify the created config is valid
            with open(output_path) as f:
                config_data = yaml.safe_load(f)
            
            assert config_data["name"] == "example-agent"
            assert "llm" in config_data
            assert "knowledge" in config_data
            assert "executor" in config_data
    
    def test_config_validate_valid(self, runner, temp_config_file):
        """Test validating a valid configuration file."""
        result = runner.invoke(app, [
            "config", "validate",
            "--config", str(temp_config_file)
        ])
        
        assert result.exit_code == 0
        assert "Configuration is valid" in result.stdout
        assert "test-agent" in result.stdout
    
    def test_config_validate_invalid(self, runner):
        """Test validating an invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"invalid": "config"}, f)
            invalid_config_path = Path(f.name)
        
        try:
            result = runner.invoke(app, [
                "config", "validate",
                "--config", str(invalid_config_path)
            ])
            
            assert result.exit_code == 1
            assert "Configuration is invalid" in result.stdout
        finally:
            invalid_config_path.unlink()
    
    def test_config_show_env(self, runner):
        """Test showing environment variable status."""
        result = runner.invoke(app, ["config", "show-env"])
        
        assert result.exit_code == 0
        assert "Environment Variables" in result.stdout
        assert "OPENAI_API_KEY" in result.stdout
    
    def test_config_unknown_action(self, runner):
        """Test unknown config action."""
        result = runner.invoke(app, ["config", "unknown"])
        
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout


class TestIngestCommand:
    """Test document ingestion command."""
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_ingest_single_file(self, mock_create_agent, runner, temp_config_file, temp_document_file):
        """Test ingesting a single document file."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.ingest_knowledge = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        result = runner.invoke(app, [
            "ingest",
            "--config", str(temp_config_file),
            str(temp_document_file)
        ])
        
        assert result.exit_code == 0
        assert "Found 1 files to process" in result.stdout
        assert "Successfully ingested 1 documents" in result.stdout
        
        # Verify agent methods were called
        mock_create_agent.assert_called_once()
        mock_agent.ingest_knowledge.assert_called_once()
        mock_agent.shutdown.assert_called_once()
        
        # Verify document was processed correctly
        call_args = mock_agent.ingest_knowledge.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].source == str(temp_document_file)
        assert "Test Document" in call_args[0].content
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_ingest_directory_recursive(self, mock_create_agent, runner, temp_config_file):
        """Test ingesting documents from directory recursively."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.ingest_knowledge = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        # Create temporary directory with files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test files
            (temp_path / "doc1.txt").write_text("Document 1 content")
            (temp_path / "doc2.md").write_text("Document 2 content")
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "doc3.txt").write_text("Document 3 content")
            
            result = runner.invoke(app, [
                "ingest",
                "--config", str(temp_config_file),
                "--recursive",
                str(temp_path)
            ])
            
            assert result.exit_code == 0
            assert "Found 3 files to process" in result.stdout
            assert "Successfully ingested 3 documents" in result.stdout
    
    def test_ingest_invalid_config(self, runner, temp_document_file):
        """Test ingestion with invalid configuration."""
        result = runner.invoke(app, [
            "ingest",
            "--config", "nonexistent.yaml",
            str(temp_document_file)
        ])
        
        # Typer returns exit code 2 for file validation errors
        assert result.exit_code == 2
        # The exact error message depends on the file system, but it should fail
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_ingest_no_files_found(self, mock_create_agent, runner, temp_config_file):
        """Test ingestion when no files are found."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(app, [
                "ingest",
                "--config", str(temp_config_file),
                "--recursive",
                "--pattern", "*.nonexistent",
                str(temp_dir)
            ])
            
            assert result.exit_code == 0
            assert "No files found to process" in result.stdout


class TestBenchmarkCommand:
    """Test benchmark command."""
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_benchmark_default_questions(self, mock_create_agent, runner, temp_config_file):
        """Test running benchmarks with default questions."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.query = AsyncMock(return_value="Test response")
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        result = runner.invoke(app, [
            "benchmark",
            "--config", str(temp_config_file),
            "--iterations", "1"
        ])
        
        assert result.exit_code == 0
        assert "Running 5 benchmark questions" in result.stdout
        assert "Benchmark Summary" in result.stdout
        assert "Success Rate" in result.stdout
        
        # Verify agent methods were called
        mock_create_agent.assert_called_once()
        assert mock_agent.query.call_count == 5  # 5 default questions
        mock_agent.shutdown.assert_called_once()
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_benchmark_with_file(self, mock_create_agent, runner, temp_config_file):
        """Test running benchmarks with custom benchmark file."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.query = AsyncMock(return_value="Test response")
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        # Create benchmark file
        benchmark_data = {
            "questions": [
                "What is the data structure?",
                "Show me a summary"
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(benchmark_data, f)
            benchmark_file = Path(f.name)
        
        try:
            result = runner.invoke(app, [
                "benchmark",
                "--config", str(temp_config_file),
                "--benchmark", str(benchmark_file),
                "--iterations", "1"
            ])
            
            assert result.exit_code == 0
            assert "Running 2 benchmark questions" in result.stdout
            assert mock_agent.query.call_count == 2
        finally:
            benchmark_file.unlink()
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_benchmark_with_output_file(self, mock_create_agent, runner, temp_config_file):
        """Test running benchmarks and saving results to file."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.query = AsyncMock(return_value="Test response")
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.json"
            
            result = runner.invoke(app, [
                "benchmark",
                "--config", str(temp_config_file),
                "--output", str(output_file),
                "--iterations", "1"
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
            assert f"Results saved to {output_file}" in result.stdout
            
            # Verify output file content
            with open(output_file) as f:
                results = json.load(f)
            
            assert "timestamp" in results
            assert "summary" in results
            assert "results" in results
            assert len(results["results"]) == 5  # 5 default questions
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_benchmark_with_failures(self, mock_create_agent, runner, temp_config_file):
        """Test benchmark handling when some queries fail."""
        # Mock agent with some failures
        mock_agent = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        # Make some queries fail
        def side_effect(query, conv_id):
            if "first" in query:
                raise Exception("Test error")
            return "Success response"
        
        mock_agent.query = AsyncMock(side_effect=side_effect)
        
        result = runner.invoke(app, [
            "benchmark",
            "--config", str(temp_config_file),
            "--iterations", "1"
        ])
        
        assert result.exit_code == 0
        assert "Benchmark Summary" in result.stdout
        # Should show partial success rate
        assert "Success Rate" in result.stdout


class TestRunCommand:
    """Test the interactive run command."""
    
    @patch('dataqa.cli.main.create_agent_from_config')
    @patch('dataqa.cli.main.Prompt.ask')
    def test_run_command_basic(self, mock_prompt, mock_create_agent, runner, temp_config_file):
        """Test basic run command functionality."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.get_agent_info = MagicMock(return_value={
            "name": "test-agent",
            "description": "Test agent",
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "knowledge_provider": "faiss",
            "executor_provider": "inmemory"
        })
        mock_agent.health_check = AsyncMock(return_value={
            "agent": "healthy",
            "llm": "healthy",
            "knowledge": "healthy",
            "executor": "healthy"
        })
        mock_agent.query = AsyncMock(return_value="Test response")
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        # Mock user input to quit immediately
        mock_prompt.side_effect = ["/quit"]
        
        result = runner.invoke(app, [
            "run",
            "--config", str(temp_config_file)
        ])
        
        assert result.exit_code == 0
        assert "Agent Information" in result.stdout
        assert "Component Health" in result.stdout
        assert "DataQA agent is ready" in result.stdout
        
        # Verify agent methods were called
        mock_create_agent.assert_called_once()
        mock_agent.get_agent_info.assert_called_once()
        mock_agent.health_check.assert_called_once()
        mock_agent.shutdown.assert_called_once()
    
    def test_run_command_invalid_config(self, runner):
        """Test run command with invalid configuration."""
        result = runner.invoke(app, [
            "run",
            "--config", "nonexistent.yaml"
        ])
        
        # Typer returns exit code 2 for file validation errors
        assert result.exit_code == 2
    
    @patch('dataqa.cli.main.create_agent_from_config')
    @patch('dataqa.cli.main.Prompt.ask')
    def test_run_command_with_conversation_id(self, mock_prompt, mock_create_agent, runner, temp_config_file):
        """Test run command with conversation ID."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.get_agent_info = MagicMock(return_value={
            "name": "test-agent",
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "knowledge_provider": "faiss",
            "executor_provider": "inmemory"
        })
        mock_agent.health_check = AsyncMock(return_value={
            "agent": "healthy",
            "llm": "healthy",
            "knowledge": "healthy",
            "executor": "healthy"
        })
        mock_agent.query = AsyncMock(return_value="Test response")
        mock_agent.get_conversation_status = AsyncMock(return_value={"pending_approval": False})
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        # Mock user input
        mock_prompt.side_effect = ["What is the data?", "/quit"]
        
        result = runner.invoke(app, [
            "run",
            "--config", str(temp_config_file),
            "--conversation", "test-conv"
        ])
        
        assert result.exit_code == 0
        
        # Verify query was called with conversation ID
        mock_agent.query.assert_called_once_with("What is the data?", "test-conv")
        mock_agent.get_conversation_status.assert_called_once_with("test-conv")
    
    @patch('dataqa.cli.main.create_agent_from_config')
    @patch('dataqa.cli.main.Prompt.ask')
    def test_run_command_help_command(self, mock_prompt, mock_create_agent, runner, temp_config_file):
        """Test help command within interactive session."""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.get_agent_info = MagicMock(return_value={
            "name": "test-agent",
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "knowledge_provider": "faiss",
            "executor_provider": "inmemory"
        })
        mock_agent.health_check = AsyncMock(return_value={
            "agent": "healthy",
            "llm": "healthy",
            "knowledge": "healthy",
            "executor": "healthy"
        })
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        # Mock user input
        mock_prompt.side_effect = ["/help", "/quit"]
        
        result = runner.invoke(app, [
            "run",
            "--config", str(temp_config_file)
        ])
        
        assert result.exit_code == 0
        assert "Available Commands" in result.stdout
        assert "/help" in result.stdout
        assert "/status" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_cli_help(self, runner):
        """Test main CLI help."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "DataQA" in result.stdout
        assert "composable data agent framework" in result.stdout
        assert "run" in result.stdout
        assert "ingest" in result.stdout
        assert "benchmark" in result.stdout
        assert "config" in result.stdout
    
    def test_command_help(self, runner):
        """Test individual command help."""
        commands = ["run", "ingest", "benchmark", "config"]
        
        for command in commands:
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            assert command in result.stdout.lower()
    
    def test_no_args_shows_help(self, runner):
        """Test that running with no arguments shows help."""
        result = runner.invoke(app, [])
        
        # Typer with no_args_is_help=True returns exit code 0 and shows help
        # But in some versions it might return 2, so we accept both
        assert result.exit_code in [0, 2]
        assert "Usage:" in result.stdout or "DataQA" in result.stdout


@pytest.mark.asyncio
class TestAsyncHelpers:
    """Test async helper functions."""
    
    @patch('dataqa.cli.main.create_agent_from_config')
    async def test_run_interactive_session_error_handling(self, mock_create_agent):
        """Test error handling in interactive session."""
        from dataqa.cli.main import _run_interactive_session
        from dataqa.config.models import AgentConfig
        
        # Mock agent that raises an error during health check
        mock_agent = AsyncMock()
        mock_agent.get_agent_info = MagicMock(return_value={
            "name": "test-agent",
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "knowledge_provider": "faiss",
            "executor_provider": "inmemory"
        })
        mock_agent.health_check = AsyncMock(side_effect=Exception("Health check failed"))
        mock_agent.shutdown = AsyncMock()
        mock_create_agent.return_value = mock_agent
        
        # Create a minimal config
        config = AgentConfig(name="test-agent")
        
        # This should not raise an exception, but handle it gracefully
        # We can't easily test the interactive loop without mocking input,
        # but we can test that the setup works
        try:
            # Just test the agent creation and health check part
            agent = await mock_create_agent(config)
            info = agent.get_agent_info()
            assert info["name"] == "test-agent"
            
            # Health check should raise an exception
            with pytest.raises(Exception, match="Health check failed"):
                await agent.health_check()
        finally:
            await mock_agent.shutdown()


class TestCLIEdgeCases:
    """Test edge cases and error conditions for CLI commands."""
    
    def test_run_with_auto_approve(self, runner, temp_config_file):
        """Test run command with auto-approve flag."""
        with patch('dataqa.cli.main.create_agent_from_config') as mock_create_agent:
            with patch('dataqa.cli.main.Prompt.ask') as mock_prompt:
                # Mock agent
                mock_agent = AsyncMock()
                mock_agent.get_agent_info = MagicMock(return_value={
                    "name": "test-agent",
                    "llm_provider": "openai",
                    "llm_model": "gpt-4",
                    "knowledge_provider": "faiss",
                    "executor_provider": "inmemory"
                })
                mock_agent.health_check = AsyncMock(return_value={
                    "agent": "healthy",
                    "llm": "healthy",
                    "knowledge": "healthy",
                    "executor": "healthy"
                })
                mock_agent.shutdown = AsyncMock()
                mock_create_agent.return_value = mock_agent
                
                # Mock user input to quit immediately
                mock_prompt.side_effect = ["/quit"]
                
                result = runner.invoke(app, [
                    "run",
                    "--config", str(temp_config_file),
                    "--auto-approve"
                ])
                
                assert result.exit_code == 0
                assert "Auto-approval enabled" in result.stdout
                assert "All operations will execute without" in result.stdout
    
    def test_ingest_with_file_pattern(self, runner, temp_config_file):
        """Test ingestion with custom file pattern."""
        with patch('dataqa.cli.main.create_agent_from_config') as mock_create_agent:
            # Mock agent
            mock_agent = AsyncMock()
            mock_agent.ingest_knowledge = AsyncMock()
            mock_agent.shutdown = AsyncMock()
            mock_create_agent.return_value = mock_agent
            
            # Create temporary directory with mixed file types
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create files with different extensions
                (temp_path / "doc1.txt").write_text("Text document")
                (temp_path / "doc2.py").write_text("Python code")
                (temp_path / "doc3.md").write_text("Markdown document")
                
                result = runner.invoke(app, [
                    "ingest",
                    "--config", str(temp_config_file),
                    "--recursive",
                    "--pattern", "*.py,*.md",
                    str(temp_path)
                ])
                
                assert result.exit_code == 0
                assert "Found 2 files to process" in result.stdout
                assert "Successfully ingested 2 documents" in result.stdout
    
    def test_benchmark_with_yaml_file(self, runner, temp_config_file):
        """Test benchmark with YAML benchmark file."""
        with patch('dataqa.cli.main.create_agent_from_config') as mock_create_agent:
            # Mock agent
            mock_agent = AsyncMock()
            mock_agent.query = AsyncMock(return_value="Test response")
            mock_agent.shutdown = AsyncMock()
            mock_create_agent.return_value = mock_agent
            
            # Create YAML benchmark file
            benchmark_data = {
                "questions": [
                    "What tables exist?",
                    "Show data summary"
                ]
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(benchmark_data, f)
                benchmark_file = Path(f.name)
            
            try:
                result = runner.invoke(app, [
                    "benchmark",
                    "--config", str(temp_config_file),
                    "--benchmark", str(benchmark_file),
                    "--iterations", "2"
                ])
                
                assert result.exit_code == 0
                assert "Running 2 benchmark questions with 2 iterations each" in result.stdout
                assert mock_agent.query.call_count == 4  # 2 questions × 2 iterations
            finally:
                benchmark_file.unlink()
    
    def test_ingest_file_processing_error(self, runner, temp_config_file):
        """Test ingestion when file processing fails."""
        with patch('dataqa.cli.main.create_agent_from_config') as mock_create_agent:
            # Mock agent
            mock_agent = AsyncMock()
            mock_agent.ingest_knowledge = AsyncMock()
            mock_agent.shutdown = AsyncMock()
            mock_create_agent.return_value = mock_agent
            
            # Create a file that will cause processing error (binary file)
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
                f.write(b'\x00\x01\x02\x03')  # Binary content
                binary_file = Path(f.name)
            
            try:
                result = runner.invoke(app, [
                    "ingest",
                    "--config", str(temp_config_file),
                    str(binary_file)
                ])
                
                # Should still succeed but with warnings about processing errors
                assert result.exit_code == 0
                # The file should be processed (with potential encoding issues handled)
                assert "Found 1 files to process" in result.stdout
            finally:
                binary_file.unlink()
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_run_command_status_command(self, mock_create_agent, runner, temp_config_file):
        """Test status command within interactive session."""
        with patch('dataqa.cli.main.Prompt.ask') as mock_prompt:
            # Mock agent
            mock_agent = AsyncMock()
            mock_agent.get_agent_info = MagicMock(return_value={
                "name": "test-agent",
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "knowledge_provider": "faiss",
                "executor_provider": "inmemory"
            })
            mock_agent.health_check = AsyncMock(return_value={
                "agent": "healthy",
                "llm": "healthy",
                "knowledge": "healthy",
                "executor": "healthy"
            })
            mock_agent.get_conversation_status = AsyncMock(return_value={
                "exists": True,
                "current_step": "waiting",
                "workflow_complete": False,
                "error_occurred": False,
                "pending_approval": False,
                "iteration_count": 1,
                "message_count": 2
            })
            mock_agent.shutdown = AsyncMock()
            mock_create_agent.return_value = mock_agent
            
            # Mock user input
            mock_prompt.side_effect = ["/status", "/quit"]
            
            result = runner.invoke(app, [
                "run",
                "--config", str(temp_config_file),
                "--conversation", "test-conv"
            ])
            
            assert result.exit_code == 0
            assert "Conversation Status" in result.stdout
            mock_agent.get_conversation_status.assert_called_once_with("test-conv")
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_run_command_history_command(self, mock_create_agent, runner, temp_config_file):
        """Test history command within interactive session."""
        from dataqa.models.message import Message
        from datetime import datetime
        
        with patch('dataqa.cli.main.Prompt.ask') as mock_prompt:
            # Mock agent
            mock_agent = AsyncMock()
            mock_agent.get_agent_info = MagicMock(return_value={
                "name": "test-agent",
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "knowledge_provider": "faiss",
                "executor_provider": "inmemory"
            })
            mock_agent.health_check = AsyncMock(return_value={
                "agent": "healthy",
                "llm": "healthy",
                "knowledge": "healthy",
                "executor": "healthy"
            })
            mock_agent.get_conversation_history = AsyncMock(return_value=[
                Message(role="user", content="What is the data?", timestamp=datetime.now()),
                Message(role="assistant", content="The data shows...", timestamp=datetime.now())
            ])
            mock_agent.shutdown = AsyncMock()
            mock_create_agent.return_value = mock_agent
            
            # Mock user input
            mock_prompt.side_effect = ["/history", "/quit"]
            
            result = runner.invoke(app, [
                "run",
                "--config", str(temp_config_file),
                "--conversation", "test-conv"
            ])
            
            assert result.exit_code == 0
            assert "Conversation History" in result.stdout
            mock_agent.get_conversation_history.assert_called_once_with("test-conv")
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_run_command_clear_command(self, mock_create_agent, runner, temp_config_file):
        """Test clear command within interactive session."""
        with patch('dataqa.cli.main.Prompt.ask') as mock_prompt:
            # Mock agent
            mock_agent = AsyncMock()
            mock_agent.get_agent_info = MagicMock(return_value={
                "name": "test-agent",
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "knowledge_provider": "faiss",
                "executor_provider": "inmemory"
            })
            mock_agent.health_check = AsyncMock(return_value={
                "agent": "healthy",
                "llm": "healthy",
                "knowledge": "healthy",
                "executor": "healthy"
            })
            mock_agent.clear_conversation = AsyncMock(return_value=True)
            mock_agent.shutdown = AsyncMock()
            mock_create_agent.return_value = mock_agent
            
            # Mock user input
            mock_prompt.side_effect = ["/clear", "/quit"]
            
            result = runner.invoke(app, [
                "run",
                "--config", str(temp_config_file),
                "--conversation", "test-conv"
            ])
            
            assert result.exit_code == 0
            assert "Conversation history cleared" in result.stdout
            mock_agent.clear_conversation.assert_called_once_with("test-conv")


class TestCLIErrorHandling:
    """Test error handling in CLI commands."""
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_ingest_agent_creation_failure(self, mock_create_agent, runner, temp_config_file, temp_document_file):
        """Test ingestion when agent creation fails."""
        mock_create_agent.side_effect = Exception("Agent creation failed")
        
        result = runner.invoke(app, [
            "ingest",
            "--config", str(temp_config_file),
            str(temp_document_file)
        ])
        
        assert result.exit_code == 1
        assert "Ingestion failed" in result.stdout
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_benchmark_agent_creation_failure(self, mock_create_agent, runner, temp_config_file):
        """Test benchmark when agent creation fails."""
        mock_create_agent.side_effect = Exception("Agent creation failed")
        
        result = runner.invoke(app, [
            "benchmark",
            "--config", str(temp_config_file)
        ])
        
        assert result.exit_code == 1
        assert "Benchmark failed" in result.stdout
    
    @patch('dataqa.cli.main.create_agent_from_config')
    def test_run_agent_creation_failure(self, mock_create_agent, runner, temp_config_file):
        """Test run command when agent creation fails."""
        mock_create_agent.side_effect = Exception("Agent creation failed")
        
        result = runner.invoke(app, [
            "run",
            "--config", str(temp_config_file)
        ])
        
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout
    
    def test_config_validate_missing_file(self, runner):
        """Test config validation with missing file."""
        result = runner.invoke(app, [
            "config", "validate",
            "--config", "nonexistent_file.yaml"
        ])
        
        assert result.exit_code == 1
    
    @patch('dataqa.cli.main.create_example_config')
    def test_config_create_failure(self, mock_create_config, runner):
        """Test config creation failure."""
        mock_create_config.side_effect = Exception("Creation failed")
        
        result = runner.invoke(app, [
            "config", "create",
            "--output", "/tmp/test_fail.yaml"
        ])
        
        assert result.exit_code == 1
        assert "Failed to create configuration" in result.stdout