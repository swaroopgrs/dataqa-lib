"""
Performance benchmarks for DataQA framework.

These tests measure query processing performance, memory usage, and resource
consumption under various scenarios and loads.
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import pandas as pd
import json

from src.dataqa.agent.agent import DataAgent
from src.dataqa.config.models import (
    AgentConfig, LLMConfig, KnowledgeConfig, ExecutorConfig, WorkflowConfig,
    LLMProvider, KnowledgeProvider, ExecutorProvider
)
from src.dataqa.models.document import Document
from src.dataqa.primitives.in_memory_executor import InMemoryExecutor
from src.dataqa.primitives.faiss_knowledge import FAISSKnowledge


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    operation: str
    duration_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    peak_memory_mb: float
    success: bool
    error: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """Monitor system resources during test execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring and return metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring capability."""
    return PerformanceMonitor()


@pytest.fixture
def large_dataset():
    """Create a large dataset for performance testing."""
    size = 10000
    return pd.DataFrame({
        'id': range(1, size + 1),
        'product': [f'Product_{i%100}' for i in range(1, size + 1)],
        'category': [f'Category_{i%20}' for i in range(1, size + 1)],
        'subcategory': [f'SubCat_{i%50}' for i in range(1, size + 1)],
        'revenue': [1000 + (i * 10) + (i % 17 * 50) for i in range(1, size + 1)],
        'quantity': [1 + (i % 100) for i in range(1, size + 1)],
        'cost': [500 + (i * 5) + (i % 13 * 25) for i in range(1, size + 1)],
        'date': pd.date_range('2020-01-01', periods=size, freq='H'),
        'region': [f'Region_{i%10}' for i in range(1, size + 1)],
        'sales_rep': [f'Rep_{i%50}' for i in range(1, size + 1)]
    })


@pytest.fixture
def large_knowledge_base():
    """Create a large knowledge base for performance testing."""
    documents = []
    
    # Schema documents
    for i in range(20):
        documents.append(Document(
            content=f"Table_{i} contains business data with columns: id, name, value, date, category. Used for analysis type {i%5}.",
            source=f"schema_table_{i}",
            metadata={"table": f"table_{i}", "type": "schema"}
        ))
    
    # Business rule documents
    for i in range(50):
        documents.append(Document(
            content=f"Business rule {i}: When analyzing data type {i%10}, consider factors A, B, and C. Apply filters based on date ranges and category restrictions.",
            source=f"business_rules_{i}",
            metadata={"rule_id": i, "type": "business_rule"}
        ))
    
    # Analysis guidance documents
    for i in range(30):
        documents.append(Document(
            content=f"Analysis guidance {i}: For metric type {i%8}, use aggregation method {i%4}. Consider seasonal adjustments and regional variations.",
            source=f"analysis_guide_{i}",
            metadata={"guide_id": i, "type": "analysis_guidance"}
        ))
    
    return documents


async def measure_performance(operation_name: str, operation_func, monitor: PerformanceMonitor) -> PerformanceMetrics:
    """Measure performance of an async operation."""
    # Force garbage collection before measurement
    gc.collect()
    
    # Get initial memory
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Start monitoring
    monitor.start_monitoring()
    start_time = time.time()
    
    try:
        # Execute operation
        result = await operation_func()
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    # Stop monitoring
    end_time = time.time()
    metrics = monitor.stop_monitoring()
    
    # Calculate final metrics
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    duration = end_time - start_time
    
    if metrics:
        peak_memory = max(m['memory_mb'] for m in metrics)
        avg_cpu = sum(m['cpu_percent'] for m in metrics) / len(metrics)
    else:
        peak_memory = final_memory
        avg_cpu = 0.0
    
    return PerformanceMetrics(
        operation=operation_name,
        duration_seconds=duration,
        memory_usage_mb=final_memory - initial_memory,
        cpu_percent=avg_cpu,
        peak_memory_mb=peak_memory,
        success=success,
        error=error,
        additional_metrics={'result_available': result is not None}
    )


class TestQueryProcessingPerformance:
    """Test query processing performance under various conditions."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_simple_query_performance(self, large_dataset, performance_monitor):
        """Test performance of simple queries."""
        # Setup executor with large dataset
        executor = InMemoryExecutor({"database_type": "duckdb"})
        await executor.load_dataframe("sales", large_dataset)
        
        # Mock LLM for consistent testing
        from unittest.mock import AsyncMock
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "data_retrieval",
            "entities": ["sales"],
            "complexity": "low",
            "requires_clarification": False,
            "suggested_approach": "Generate simple SQL query"
        }
        mock_llm.generate_code.return_value = "SELECT COUNT(*) FROM sales"
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True, "issues": [], "security_concerns": [], "suggestions": [], "risk_level": "low"
        }
        mock_llm.format_response.return_value = "Query completed successfully."
        
        # Create minimal knowledge base
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "perf_index")
            )
            await knowledge.ingest([Document(content="Test doc", source="test", metadata={})])
            
            config = AgentConfig(
                name="perf-test-agent",
                description="Performance test agent",
                workflow=WorkflowConfig(require_approval=False)
            )
            
            agent = DataAgent(config=config, llm=mock_llm, knowledge=knowledge, executor=executor)
            
            # Measure performance
            async def query_operation():
                return await agent.query("Count the number of sales records")
            
            metrics = await measure_performance("simple_query", query_operation, performance_monitor)
            
            # Performance assertions
            assert metrics.success, f"Query failed: {metrics.error}"
            assert metrics.duration_seconds < 5.0, f"Query took too long: {metrics.duration_seconds}s"
            assert metrics.memory_usage_mb < 100, f"Memory usage too high: {metrics.memory_usage_mb}MB"
            
            await agent.shutdown()
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_complex_query_performance(self, large_dataset, performance_monitor):
        """Test performance of complex analytical queries."""
        executor = InMemoryExecutor({"database_type": "duckdb"})
        await executor.load_dataframe("sales", large_dataset)
        
        from unittest.mock import AsyncMock
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "complex_analysis",
            "entities": ["sales", "revenue", "trends"],
            "complexity": "high",
            "requires_clarification": False,
            "suggested_approach": "Generate complex analytical SQL"
        }
        
        # Complex aggregation query
        mock_llm.generate_code.return_value = """
        SELECT 
            region,
            category,
            DATE_TRUNC('month', date) as month,
            SUM(revenue) as total_revenue,
            AVG(revenue) as avg_revenue,
            COUNT(*) as transaction_count,
            SUM(quantity) as total_quantity,
            (SUM(revenue) - SUM(cost)) as profit
        FROM sales 
        GROUP BY region, category, DATE_TRUNC('month', date)
        HAVING SUM(revenue) > 1000
        ORDER BY total_revenue DESC
        LIMIT 100
        """
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True, "issues": [], "security_concerns": [], "suggestions": [], "risk_level": "low"
        }
        mock_llm.format_response.return_value = "Complex analysis completed."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "complex_index")
            )
            await knowledge.ingest([Document(content="Test doc", source="test", metadata={})])
            
            config = AgentConfig(
                name="complex-perf-agent",
                description="Complex performance test agent",
                workflow=WorkflowConfig(require_approval=False)
            )
            
            agent = DataAgent(config=config, llm=mock_llm, knowledge=knowledge, executor=executor)
            
            async def complex_query_operation():
                return await agent.query("Analyze revenue trends by region and category with profit calculations")
            
            metrics = await measure_performance("complex_query", complex_query_operation, performance_monitor)
            
            # Performance assertions for complex queries
            assert metrics.success, f"Complex query failed: {metrics.error}"
            assert metrics.duration_seconds < 15.0, f"Complex query took too long: {metrics.duration_seconds}s"
            assert metrics.memory_usage_mb < 200, f"Memory usage too high: {metrics.memory_usage_mb}MB"
            
            await agent.shutdown()
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_query_performance(self, large_dataset, performance_monitor):
        """Test performance under concurrent query load."""
        executor = InMemoryExecutor({"database_type": "duckdb"})
        await executor.load_dataframe("sales", large_dataset)
        
        from unittest.mock import AsyncMock
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "data_retrieval",
            "entities": ["sales"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate SQL query"
        }
        
        # Different queries for concurrent testing
        queries_and_sql = [
            ("Count sales by region", "SELECT region, COUNT(*) FROM sales GROUP BY region"),
            ("Average revenue by category", "SELECT category, AVG(revenue) FROM sales GROUP BY category"),
            ("Top products by quantity", "SELECT product, SUM(quantity) FROM sales GROUP BY product ORDER BY SUM(quantity) DESC LIMIT 10"),
            ("Monthly revenue trends", "SELECT DATE_TRUNC('month', date) as month, SUM(revenue) FROM sales GROUP BY month ORDER BY month"),
            ("Sales rep performance", "SELECT sales_rep, SUM(revenue), COUNT(*) FROM sales GROUP BY sales_rep ORDER BY SUM(revenue) DESC")
        ]
        
        mock_llm.generate_code.side_effect = [sql for _, sql in queries_and_sql]
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True, "issues": [], "security_concerns": [], "suggestions": [], "risk_level": "low"
        }
        mock_llm.format_response.return_value = "Query completed."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "concurrent_index")
            )
            await knowledge.ingest([Document(content="Test doc", source="test", metadata={})])
            
            config = AgentConfig(
                name="concurrent-perf-agent",
                description="Concurrent performance test agent",
                workflow=WorkflowConfig(require_approval=False)
            )
            
            agent = DataAgent(config=config, llm=mock_llm, knowledge=knowledge, executor=executor)
            
            async def concurrent_queries_operation():
                # Execute queries concurrently
                tasks = []
                for i, (query, _) in enumerate(queries_and_sql):
                    task = agent.query(query, f"concurrent_{i}")
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            metrics = await measure_performance("concurrent_queries", concurrent_queries_operation, performance_monitor)
            
            # Performance assertions for concurrent queries
            assert metrics.success, f"Concurrent queries failed: {metrics.error}"
            assert metrics.duration_seconds < 20.0, f"Concurrent queries took too long: {metrics.duration_seconds}s"
            assert metrics.memory_usage_mb < 300, f"Memory usage too high: {metrics.memory_usage_mb}MB"
            
            await agent.shutdown()


class TestKnowledgeBasePerformance:
    """Test knowledge base performance with large document sets."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_knowledge_ingestion_performance(self, large_knowledge_base, performance_monitor):
        """Test performance of ingesting large knowledge bases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "ingestion_index")
            )
            
            async def ingestion_operation():
                await knowledge.ingest(large_knowledge_base)
                return len(large_knowledge_base)
            
            metrics = await measure_performance("knowledge_ingestion", ingestion_operation, performance_monitor)
            
            # Performance assertions
            assert metrics.success, f"Knowledge ingestion failed: {metrics.error}"
            assert metrics.duration_seconds < 60.0, f"Ingestion took too long: {metrics.duration_seconds}s"
            assert metrics.memory_usage_mb < 500, f"Memory usage too high: {metrics.memory_usage_mb}MB"
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_knowledge_search_performance(self, large_knowledge_base, performance_monitor):
        """Test performance of searching large knowledge bases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "search_index")
            )
            
            # Ingest knowledge base first
            await knowledge.ingest(large_knowledge_base)
            
            search_queries = [
                "business rules for data analysis",
                "table schema information",
                "analysis guidance for metrics",
                "category restrictions and filters",
                "seasonal adjustments in analysis"
            ]
            
            async def search_operation():
                results = []
                for query in search_queries:
                    search_results = await knowledge.search(query, limit=5)
                    results.extend(search_results)
                return results
            
            metrics = await measure_performance("knowledge_search", search_operation, performance_monitor)
            
            # Performance assertions
            assert metrics.success, f"Knowledge search failed: {metrics.error}"
            assert metrics.duration_seconds < 10.0, f"Search took too long: {metrics.duration_seconds}s"
            assert metrics.memory_usage_mb < 100, f"Memory usage too high: {metrics.memory_usage_mb}MB"


class TestMemoryUsagePatterns:
    """Test memory usage patterns and resource consumption."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_over_time(self, large_dataset, performance_monitor):
        """Test memory usage patterns over extended operation."""
        executor = InMemoryExecutor({"database_type": "duckdb"})
        await executor.load_dataframe("sales", large_dataset)
        
        from unittest.mock import AsyncMock
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "data_retrieval",
            "entities": ["sales"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate SQL query"
        }
        
        queries = [
            "SELECT COUNT(*) FROM sales",
            "SELECT AVG(revenue) FROM sales",
            "SELECT region, SUM(revenue) FROM sales GROUP BY region",
            "SELECT category, COUNT(*) FROM sales GROUP BY category",
            "SELECT product, MAX(revenue) FROM sales GROUP BY product"
        ]
        
        mock_llm.generate_code.side_effect = queries * 4  # Repeat queries
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True, "issues": [], "security_concerns": [], "suggestions": [], "risk_level": "low"
        }
        mock_llm.format_response.return_value = "Query completed."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "memory_index")
            )
            await knowledge.ingest([Document(content="Test doc", source="test", metadata={})])
            
            config = AgentConfig(
                name="memory-test-agent",
                description="Memory usage test agent",
                workflow=WorkflowConfig(require_approval=False)
            )
            
            agent = DataAgent(config=config, llm=mock_llm, knowledge=knowledge, executor=executor)
            
            async def extended_operation():
                # Execute multiple queries over time
                for i in range(20):
                    query_text = f"Execute query {i % len(queries)}"
                    await agent.query(query_text, f"memory_test_{i}")
                    
                    # Small delay to allow monitoring
                    await asyncio.sleep(0.1)
                
                return "completed"
            
            metrics = await measure_performance("extended_operation", extended_operation, performance_monitor)
            
            # Memory usage assertions
            assert metrics.success, f"Extended operation failed: {metrics.error}"
            assert metrics.memory_usage_mb < 150, f"Memory growth too high: {metrics.memory_usage_mb}MB"
            
            # Check for memory leaks (peak should not be much higher than final)
            memory_growth_ratio = metrics.memory_usage_mb / metrics.peak_memory_mb if metrics.peak_memory_mb > 0 else 0
            assert memory_growth_ratio > 0.5, f"Potential memory leak detected: {memory_growth_ratio}"
            
            await agent.shutdown()
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_conversation_memory_usage(self, large_dataset, performance_monitor):
        """Test memory usage with multiple concurrent conversations."""
        executor = InMemoryExecutor({"database_type": "duckdb"})
        await executor.load_dataframe("sales", large_dataset)
        
        from unittest.mock import AsyncMock
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "data_retrieval",
            "entities": ["sales"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate SQL query"
        }
        mock_llm.generate_code.return_value = "SELECT * FROM sales LIMIT 10"
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True, "issues": [], "security_concerns": [], "suggestions": [], "risk_level": "low"
        }
        mock_llm.format_response.return_value = "Query completed."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "conversation_index")
            )
            await knowledge.ingest([Document(content="Test doc", source="test", metadata={})])
            
            config = AgentConfig(
                name="conversation-memory-agent",
                description="Conversation memory test agent",
                workflow=WorkflowConfig(require_approval=False, conversation_memory=True)
            )
            
            agent = DataAgent(config=config, llm=mock_llm, knowledge=knowledge, executor=executor)
            
            async def conversation_operation():
                # Create multiple conversations
                num_conversations = 50
                for i in range(num_conversations):
                    conv_id = f"conv_{i}"
                    
                    # Multiple queries per conversation
                    for j in range(5):
                        query = f"Query {j} in conversation {i}"
                        await agent.query(query, conv_id)
                
                return f"Created {num_conversations} conversations"
            
            metrics = await measure_performance("conversation_memory", conversation_operation, performance_monitor)
            
            # Memory usage assertions for conversations
            assert metrics.success, f"Conversation operation failed: {metrics.error}"
            assert metrics.memory_usage_mb < 200, f"Conversation memory usage too high: {metrics.memory_usage_mb}MB"
            
            await agent.shutdown()


class TestResourceConsumption:
    """Test overall resource consumption patterns."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cpu_usage_patterns(self, large_dataset, performance_monitor):
        """Test CPU usage patterns under different workloads."""
        executor = InMemoryExecutor({"database_type": "duckdb"})
        await executor.load_dataframe("sales", large_dataset)
        
        from unittest.mock import AsyncMock
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "cpu_intensive",
            "entities": ["sales"],
            "complexity": "high",
            "requires_clarification": False,
            "suggested_approach": "Generate CPU-intensive query"
        }
        
        # CPU-intensive query
        mock_llm.generate_code.return_value = """
        SELECT 
            region,
            category,
            product,
            COUNT(*) as count,
            SUM(revenue) as total_revenue,
            AVG(revenue) as avg_revenue,
            STDDEV(revenue) as stddev_revenue,
            MIN(revenue) as min_revenue,
            MAX(revenue) as max_revenue,
            SUM(quantity) as total_quantity,
            AVG(quantity) as avg_quantity
        FROM sales 
        GROUP BY region, category, product
        HAVING COUNT(*) > 5
        ORDER BY total_revenue DESC
        """
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True, "issues": [], "security_concerns": [], "suggestions": [], "risk_level": "low"
        }
        mock_llm.format_response.return_value = "CPU-intensive analysis completed."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "cpu_index")
            )
            await knowledge.ingest([Document(content="Test doc", source="test", metadata={})])
            
            config = AgentConfig(
                name="cpu-test-agent",
                description="CPU usage test agent",
                workflow=WorkflowConfig(require_approval=False)
            )
            
            agent = DataAgent(config=config, llm=mock_llm, knowledge=knowledge, executor=executor)
            
            async def cpu_intensive_operation():
                return await agent.query("Perform comprehensive statistical analysis of all sales data")
            
            metrics = await measure_performance("cpu_intensive", cpu_intensive_operation, performance_monitor)
            
            # CPU usage assertions
            assert metrics.success, f"CPU-intensive operation failed: {metrics.error}"
            assert metrics.duration_seconds < 30.0, f"CPU operation took too long: {metrics.duration_seconds}s"
            # CPU usage can be high for intensive operations, but should be reasonable
            assert metrics.cpu_percent < 200.0, f"CPU usage too high: {metrics.cpu_percent}%"
            
            await agent.shutdown()


@pytest.mark.slow
class TestBenchmarkSuite:
    """Comprehensive benchmark suite for performance regression testing."""
    
    @pytest.mark.asyncio
    async def test_benchmark_suite(self, large_dataset, large_knowledge_base, performance_monitor):
        """Run comprehensive benchmark suite and generate report."""
        benchmark_results = []
        
        # Setup components
        executor = InMemoryExecutor({"database_type": "duckdb"})
        await executor.load_dataframe("sales", large_dataset)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "benchmark_index")
            )
            await knowledge.ingest(large_knowledge_base)
            
            from unittest.mock import AsyncMock
            mock_llm = AsyncMock()
            mock_llm.analyze_query.return_value = {
                "intent": "data analysis",
                "query_type": "benchmark",
                "entities": ["sales"],
                "complexity": "moderate",
                "requires_clarification": False,
                "suggested_approach": "Generate benchmark query"
            }
            mock_llm.validate_generated_code.return_value = {
                "is_valid": True, "issues": [], "security_concerns": [], "suggestions": [], "risk_level": "low"
            }
            mock_llm.format_response.return_value = "Benchmark query completed."
            
            config = AgentConfig(
                name="benchmark-agent",
                description="Comprehensive benchmark agent",
                workflow=WorkflowConfig(require_approval=False)
            )
            
            agent = DataAgent(config=config, llm=mock_llm, knowledge=knowledge, executor=executor)
            
            # Benchmark scenarios
            benchmark_scenarios = [
                ("Simple Count", "SELECT COUNT(*) FROM sales"),
                ("Group By Region", "SELECT region, COUNT(*) FROM sales GROUP BY region"),
                ("Revenue Analysis", "SELECT category, SUM(revenue), AVG(revenue) FROM sales GROUP BY category"),
                ("Time Series", "SELECT DATE_TRUNC('day', date) as day, SUM(revenue) FROM sales GROUP BY day ORDER BY day"),
                ("Complex Join", "SELECT region, category, SUM(revenue) FROM sales GROUP BY region, category HAVING SUM(revenue) > 10000")
            ]
            
            for scenario_name, sql_query in benchmark_scenarios:
                mock_llm.generate_code.return_value = sql_query
                
                async def benchmark_operation():
                    return await agent.query(f"Execute {scenario_name.lower()}")
                
                metrics = await measure_performance(scenario_name, benchmark_operation, performance_monitor)
                benchmark_results.append(metrics)
            
            await agent.shutdown()
        
        # Generate benchmark report
        report = {
            "timestamp": time.time(),
            "total_scenarios": len(benchmark_results),
            "successful_scenarios": sum(1 for r in benchmark_results if r.success),
            "average_duration": sum(r.duration_seconds for r in benchmark_results) / len(benchmark_results),
            "average_memory_usage": sum(r.memory_usage_mb for r in benchmark_results) / len(benchmark_results),
            "peak_memory_usage": max(r.peak_memory_mb for r in benchmark_results),
            "scenarios": [
                {
                    "name": r.operation,
                    "duration_seconds": r.duration_seconds,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_percent": r.cpu_percent,
                    "success": r.success,
                    "error": r.error
                }
                for r in benchmark_results
            ]
        }
        
        # Save benchmark report
        report_path = Path("benchmark_results.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Performance assertions for overall benchmark
        assert report["successful_scenarios"] == report["total_scenarios"], "Some benchmark scenarios failed"
        assert report["average_duration"] < 10.0, f"Average duration too high: {report['average_duration']}s"
        assert report["average_memory_usage"] < 100.0, f"Average memory usage too high: {report['average_memory_usage']}MB"
        
        print(f"\nBenchmark Report Summary:")
        print(f"Total Scenarios: {report['total_scenarios']}")
        print(f"Successful: {report['successful_scenarios']}")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print(f"Average Memory Usage: {report['average_memory_usage']:.2f}MB")
        print(f"Peak Memory Usage: {report['peak_memory_usage']:.2f}MB")
        print(f"Report saved to: {report_path}")