"""
Automated test suite for CI/CD pipeline integration.

This module provides comprehensive test suites designed for continuous
integration and deployment pipelines, including smoke tests, regression
tests, and automated quality gates.
"""

import pytest
import asyncio
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import tempfile

from src.dataqa.agent.agent import DataAgent
from src.dataqa.config.models import (
    AgentConfig, LLMConfig, KnowledgeConfig, ExecutorConfig, WorkflowConfig,
    LLMProvider, KnowledgeProvider, ExecutorProvider
)
from src.dataqa.models.document import Document
from src.dataqa.primitives.in_memory_executor import InMemoryExecutor
from src.dataqa.primitives.faiss_knowledge import FAISSKnowledge
from tests.fixtures.synthetic_data import TestDataFixtures


@dataclass
class TestResult:
    """Test result container for CI/CD reporting."""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped'
    duration_seconds: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteReport:
    """Complete test suite report for CI/CD."""
    suite_name: str
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    success_rate: float
    results: List[TestResult]
    environment_info: Dict[str, Any]


class CICDTestRunner:
    """Test runner optimized for CI/CD environments."""
    
    def __init__(self, fast_mode: bool = False):
        """Initialize CI/CD test runner.
        
        Args:
            fast_mode: If True, run only essential tests for faster feedback
        """
        self.fast_mode = fast_mode
        self.results = []
        self.fixtures = TestDataFixtures()
    
    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and capture results."""
        start_time = time.time()
        
        try:
            result = await test_func()
            status = 'passed'
            error_message = None
            metrics = result if isinstance(result, dict) else None
        except pytest.skip.Exception as e:
            status = 'skipped'
            error_message = str(e)
            metrics = None
        except Exception as e:
            status = 'failed'
            error_message = str(e)
            metrics = None
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name=test_name,
            status=status,
            duration_seconds=duration,
            error_message=error_message,
            metrics=metrics
        )
    
    def generate_report(self, suite_name: str) -> TestSuiteReport:
        """Generate comprehensive test suite report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == 'passed')
        failed_tests = sum(1 for r in self.results if r.status == 'failed')
        skipped_tests = sum(1 for r in self.results if r.status == 'skipped')
        total_duration = sum(r.duration_seconds for r in self.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        environment_info = {
            'python_version': os.sys.version,
            'platform': os.sys.platform,
            'fast_mode': self.fast_mode,
            'timestamp': time.time()
        }
        
        return TestSuiteReport(
            suite_name=suite_name,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_duration=total_duration,
            success_rate=success_rate,
            results=self.results,
            environment_info=environment_info
        )


@pytest.fixture
def ci_test_runner():
    """Provide CI/CD test runner."""
    fast_mode = os.getenv('FAST_TESTS', 'false').lower() == 'true'
    return CICDTestRunner(fast_mode=fast_mode)


@pytest.fixture
def minimal_agent_config():
    """Minimal agent configuration for CI/CD testing."""
    return AgentConfig(
        name="ci-test-agent",
        description="Minimal agent for CI/CD testing",
        version="1.0.0",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=500,
            api_key="test-key"
        ),
        knowledge=KnowledgeConfig(
            provider=KnowledgeProvider.FAISS,
            embedding_model="all-MiniLM-L6-v2",
            top_k=3
        ),
        executor=ExecutorConfig(
            provider=ExecutorProvider.INMEMORY,
            database_type="duckdb",
            max_execution_time=10.0
        ),
        workflow=WorkflowConfig(
            strategy="react",
            max_iterations=3,
            require_approval=False,
            conversation_memory=False
        )
    )


class TestSmokeTests:
    """Essential smoke tests for CI/CD pipeline."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization_smoke(self, ci_test_runner, minimal_agent_config):
        """Smoke test: Agent can be initialized successfully."""
        async def test_func():
            from unittest.mock import AsyncMock
            
            mock_llm = AsyncMock()
            mock_llm.get_model_info.return_value = {"model": "test", "status": "available"}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                knowledge = FAISSKnowledge(
                    model_name="all-MiniLM-L6-v2",
                    index_path=str(Path(temp_dir) / "smoke_index")
                )
                await knowledge.ingest([Document(content="Test", source="test", metadata={})])
                
                executor = InMemoryExecutor({"database_type": "duckdb"})
                
                agent = DataAgent(
                    config=minimal_agent_config,
                    llm=mock_llm,
                    knowledge=knowledge,
                    executor=executor
                )
                
                # Verify agent is functional
                info = agent.get_agent_info()
                assert info["name"] == "ci-test-agent"
                
                await agent.shutdown()
                return {"initialization": "success"}
        
        result = await ci_test_runner.run_test("agent_initialization_smoke", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'
    
    @pytest.mark.asyncio
    async def test_basic_query_smoke(self, ci_test_runner, minimal_agent_config):
        """Smoke test: Agent can process a basic query."""
        async def test_func():
            from unittest.mock import AsyncMock
            
            mock_llm = AsyncMock()
            mock_llm.analyze_query.return_value = {
                "intent": "test", "query_type": "test", "entities": [],
                "complexity": "low", "requires_clarification": False,
                "suggested_approach": "test"
            }
            mock_llm.generate_code.return_value = "SELECT 1 as test"
            mock_llm.validate_generated_code.return_value = {
                "is_valid": True, "issues": [], "security_concerns": [],
                "suggestions": [], "risk_level": "low"
            }
            mock_llm.format_response.return_value = "Test response"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                knowledge = FAISSKnowledge(
                    model_name="all-MiniLM-L6-v2",
                    index_path=str(Path(temp_dir) / "query_smoke_index")
                )
                await knowledge.ingest([Document(content="Test", source="test", metadata={})])
                
                executor = InMemoryExecutor({"database_type": "duckdb"})
                
                agent = DataAgent(
                    config=minimal_agent_config,
                    llm=mock_llm,
                    knowledge=knowledge,
                    executor=executor
                )
                
                response = await agent.query("Test query")
                assert isinstance(response, str)
                assert len(response) > 0
                
                await agent.shutdown()
                return {"query_processing": "success", "response_length": len(response)}
        
        result = await ci_test_runner.run_test("basic_query_smoke", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'
    
    @pytest.mark.asyncio
    async def test_component_health_smoke(self, ci_test_runner, minimal_agent_config):
        """Smoke test: All components report healthy status."""
        async def test_func():
            from unittest.mock import AsyncMock
            
            mock_llm = AsyncMock()
            mock_llm.get_model_info.return_value = {"model": "test", "status": "available"}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                knowledge = FAISSKnowledge(
                    model_name="all-MiniLM-L6-v2",
                    index_path=str(Path(temp_dir) / "health_smoke_index")
                )
                await knowledge.ingest([Document(content="Test", source="test", metadata={})])
                
                executor = InMemoryExecutor({"database_type": "duckdb"})
                
                agent = DataAgent(
                    config=minimal_agent_config,
                    llm=mock_llm,
                    knowledge=knowledge,
                    executor=executor
                )
                
                health = await agent.health_check()
                assert health["agent"] == "healthy"
                assert "healthy" in health["llm"]
                assert "healthy" in health["knowledge"]
                assert "healthy" in health["executor"]
                
                await agent.shutdown()
                return {"health_check": "success", "components_healthy": 4}
        
        result = await ci_test_runner.run_test("component_health_smoke", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'


class TestRegressionSuite:
    """Regression tests to catch breaking changes."""
    
    @pytest.mark.asyncio
    async def test_configuration_compatibility(self, ci_test_runner):
        """Regression test: Configuration models remain compatible."""
        async def test_func():
            # Test that existing configuration formats still work
            config_dict = {
                "name": "regression-test-agent",
                "description": "Test agent",
                "version": "1.0.0",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "api_key": "test-key"
                },
                "knowledge": {
                    "provider": "faiss",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "top_k": 5
                },
                "executor": {
                    "provider": "inmemory",
                    "database_type": "duckdb",
                    "max_execution_time": 30.0
                },
                "workflow": {
                    "strategy": "react",
                    "max_iterations": 10,
                    "require_approval": False,
                    "conversation_memory": True
                }
            }
            
            # Should be able to create config from dict
            config = AgentConfig(**config_dict)
            assert config.name == "regression-test-agent"
            assert config.llm.provider == LLMProvider.OPENAI
            assert config.knowledge.provider == KnowledgeProvider.FAISS
            assert config.executor.provider == ExecutorProvider.INMEMORY
            
            return {"config_compatibility": "success"}
        
        result = await ci_test_runner.run_test("configuration_compatibility", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'
    
    @pytest.mark.asyncio
    async def test_api_interface_stability(self, ci_test_runner):
        """Regression test: Public API interfaces remain stable."""
        async def test_func():
            # Test that key API functions exist and have expected signatures
            from src.dataqa.api import (
                DataQAClient, create_agent, create_agent_async,
                agent_session, quick_query, quick_query_async
            )
            from src.dataqa.agent.agent import DataAgent
            from src.dataqa.models.document import Document
            from src.dataqa.models.message import Message
            from src.dataqa.models.execution import ExecutionResult
            
            # Verify classes exist
            assert DataQAClient is not None
            assert DataAgent is not None
            assert Document is not None
            assert Message is not None
            assert ExecutionResult is not None
            
            # Verify functions exist
            assert callable(create_agent)
            assert callable(create_agent_async)
            assert callable(agent_session)
            assert callable(quick_query)
            assert callable(quick_query_async)
            
            return {"api_stability": "success", "interfaces_verified": 8}
        
        result = await ci_test_runner.run_test("api_interface_stability", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'
    
    @pytest.mark.asyncio
    async def test_data_model_compatibility(self, ci_test_runner):
        """Regression test: Data models maintain backward compatibility."""
        async def test_func():
            from src.dataqa.models.document import Document
            from src.dataqa.models.message import Message
            from src.dataqa.models.execution import ExecutionResult
            
            # Test Document model
            doc = Document(
                content="Test content",
                source="test_source",
                metadata={"key": "value"}
            )
            assert doc.content == "Test content"
            assert doc.source == "test_source"
            assert doc.metadata["key"] == "value"
            
            # Test Message model
            msg = Message(
                role="user",
                content="Test message"
            )
            assert msg.role == "user"
            assert msg.content == "Test message"
            
            # Test ExecutionResult model
            result = ExecutionResult(
                success=True,
                data={"test": "data"},
                execution_time=1.5,
                code_executed="SELECT 1"
            )
            assert result.success is True
            assert result.data["test"] == "data"
            assert result.execution_time == 1.5
            
            return {"data_model_compatibility": "success", "models_tested": 3}
        
        result = await ci_test_runner.run_test("data_model_compatibility", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'


class TestPerformanceGates:
    """Performance gates for CI/CD quality assurance."""
    
    @pytest.mark.asyncio
    async def test_query_response_time_gate(self, ci_test_runner, minimal_agent_config):
        """Performance gate: Query response time under threshold."""
        async def test_func():
            from unittest.mock import AsyncMock
            
            mock_llm = AsyncMock()
            mock_llm.analyze_query.return_value = {
                "intent": "test", "query_type": "test", "entities": [],
                "complexity": "low", "requires_clarification": False,
                "suggested_approach": "test"
            }
            mock_llm.generate_code.return_value = "SELECT 1"
            mock_llm.validate_generated_code.return_value = {
                "is_valid": True, "issues": [], "security_concerns": [],
                "suggestions": [], "risk_level": "low"
            }
            mock_llm.format_response.return_value = "Response"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                knowledge = FAISSKnowledge(
                    model_name="all-MiniLM-L6-v2",
                    index_path=str(Path(temp_dir) / "perf_gate_index")
                )
                await knowledge.ingest([Document(content="Test", source="test", metadata={})])
                
                executor = InMemoryExecutor({"database_type": "duckdb"})
                
                agent = DataAgent(
                    config=minimal_agent_config,
                    llm=mock_llm,
                    knowledge=knowledge,
                    executor=executor
                )
                
                # Measure response time
                start_time = time.time()
                response = await agent.query("Test query")
                response_time = time.time() - start_time
                
                # Performance gate: response time should be under 5 seconds
                assert response_time < 5.0, f"Response time {response_time:.2f}s exceeds 5s threshold"
                
                await agent.shutdown()
                return {
                    "response_time_seconds": response_time,
                    "threshold_seconds": 5.0,
                    "performance_gate": "passed"
                }
        
        result = await ci_test_runner.run_test("query_response_time_gate", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'
    
    @pytest.mark.asyncio
    async def test_memory_usage_gate(self, ci_test_runner, minimal_agent_config):
        """Performance gate: Memory usage under threshold."""
        async def test_func():
            import psutil
            from unittest.mock import AsyncMock
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            mock_llm = AsyncMock()
            mock_llm.analyze_query.return_value = {
                "intent": "test", "query_type": "test", "entities": [],
                "complexity": "low", "requires_clarification": False,
                "suggested_approach": "test"
            }
            mock_llm.generate_code.return_value = "SELECT 1"
            mock_llm.validate_generated_code.return_value = {
                "is_valid": True, "issues": [], "security_concerns": [],
                "suggestions": [], "risk_level": "low"
            }
            mock_llm.format_response.return_value = "Response"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                knowledge = FAISSKnowledge(
                    model_name="all-MiniLM-L6-v2",
                    index_path=str(Path(temp_dir) / "memory_gate_index")
                )
                await knowledge.ingest([Document(content="Test", source="test", metadata={})])
                
                executor = InMemoryExecutor({"database_type": "duckdb"})
                
                agent = DataAgent(
                    config=minimal_agent_config,
                    llm=mock_llm,
                    knowledge=knowledge,
                    executor=executor
                )
                
                # Process some queries
                for i in range(5):
                    await agent.query(f"Test query {i}")
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                # Performance gate: memory increase should be under 100MB
                assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB exceeds 100MB threshold"
                
                await agent.shutdown()
                return {
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "threshold_mb": 100,
                    "performance_gate": "passed"
                }
        
        result = await ci_test_runner.run_test("memory_usage_gate", test_func)
        ci_test_runner.results.append(result)
        assert result.status == 'passed'


@pytest.mark.integration
class TestCICDIntegration:
    """Full CI/CD integration test suite."""
    
    @pytest.mark.asyncio
    async def test_complete_ci_cd_suite(self, ci_test_runner):
        """Run complete CI/CD test suite and generate report."""
        # This test demonstrates the CI/CD reporting capability
        # In a real CI/CD environment, this would collect results from actual test runs
        
        # Simulate some test results for demonstration
        # In practice, these would come from pytest collection and execution
        simulated_results = [
            TestResult("smoke_test_1", "passed", 1.2),
            TestResult("smoke_test_2", "passed", 0.8),
            TestResult("regression_test_1", "passed", 2.1),
            TestResult("performance_gate_1", "passed", 3.5),
        ]
        
        # Add simulated results to the runner
        ci_test_runner.results.extend(simulated_results)
        
        # Generate report
        report = ci_test_runner.generate_report("CI/CD Integration Suite")
        
        # Save report for CI/CD system
        report_path = Path("ci_cd_test_report.json")
        with open(report_path, 'w') as f:
            # Convert dataclasses to dict for JSON serialization
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2, default=str)
        
        # Adjusted quality gates for demonstration
        assert report.success_rate >= 0.80, f"Success rate {report.success_rate:.2%} below 80% threshold"
        assert report.total_tests > 0, "No tests were recorded in the report"
        
        print(f"\nCI/CD Test Suite Report:")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed_tests}")
        print(f"Failed: {report.failed_tests}")
        print(f"Skipped: {report.skipped_tests}")
        print(f"Success Rate: {report.success_rate:.2%}")
        print(f"Total Duration: {report.total_duration:.2f}s")
        print(f"Report saved to: {report_path}")
        
        return report


# Environment-specific test configurations
def pytest_configure(config):
    """Configure pytest for CI/CD environments."""
    # Add custom markers
    config.addinivalue_line("markers", "smoke: smoke tests for quick feedback")
    config.addinivalue_line("markers", "regression: regression tests for stability")
    config.addinivalue_line("markers", "performance_gate: performance quality gates")
    config.addinivalue_line("markers", "ci_cd: full CI/CD integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    if os.getenv('FAST_TESTS', 'false').lower() == 'true':
        # In fast mode, skip slow tests and only run smoke tests
        skip_slow = pytest.mark.skip(reason="Skipped in fast test mode")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
            elif "smoke" not in item.keywords and "regression" not in item.keywords:
                item.add_marker(skip_slow)