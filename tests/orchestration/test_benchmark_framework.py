"""
Tests for the comprehensive benchmarking framework.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.dataqa.orchestration.evaluation.benchmark import (
    BenchmarkFramework,
    BenchmarkSuite,
    TestCase,
    GroundTruth,
    EvaluationCriterion,
    PerformanceBaseline,
    QualityThreshold,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkConfiguration
)
from src.dataqa.orchestration.models import AgentConfiguration, AgentRole, CapabilityType
from src.dataqa.config.models import LLMConfig, LLMProvider
from src.dataqa.exceptions import DataQAError


class TestBenchmarkFramework:
    """Test cases for BenchmarkFramework."""
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth data."""
        return GroundTruth(
            expected_output={"result": "success", "value": 42},
            acceptable_variations=[
                {"result": "success", "value": 41},
                {"result": "success", "value": 43}
            ],
            evaluation_criteria={"accuracy": 0.9, "completeness": 1.0}
        )
    
    @pytest.fixture
    def sample_test_case(self, sample_ground_truth):
        """Create sample test case."""
        return TestCase(
            name="Basic Calculation Test",
            description="Test basic mathematical calculation",
            category="mathematics",
            difficulty_level="easy",
            inputs={"query": "What is 6 * 7?", "context": "arithmetic"},
            ground_truth=sample_ground_truth,
            timeout_seconds=30,
            tags=["math", "basic", "calculation"]
        )
    
    @pytest.fixture
    def sample_evaluation_criteria(self):
        """Create sample evaluation criteria."""
        return [
            EvaluationCriterion(
                name="correctness",
                description="Accuracy of the response",
                weight=0.6,
                scoring_method="llm_judge"
            ),
            EvaluationCriterion(
                name="completeness",
                description="Completeness of the response",
                weight=0.4,
                scoring_method="similarity"
            )
        ]
    
    @pytest.fixture
    def sample_benchmark_suite(self, sample_test_case, sample_evaluation_criteria):
        """Create sample benchmark suite."""
        return BenchmarkSuite(
            name="Basic Math Suite",
            description="Test suite for basic mathematical operations",
            version="1.0.0",
            test_cases=[sample_test_case],
            evaluation_criteria=sample_evaluation_criteria,
            quality_thresholds={
                "min_success_rate": QualityThreshold(
                    metric_name="success_rate",
                    minimum_value=0.8,
                    description="Minimum acceptable success rate"
                )
            }
        )
    
    @pytest.fixture
    def sample_agent_config(self):
        """Create sample agent configuration."""
        return AgentConfiguration(
            name="Test Math Agent",
            role=AgentRole.WORKER,
            specialization="mathematics"
        )
    
    @pytest.fixture
    def llm_config(self):
        """Create sample LLM configuration."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            temperature=0.1
        )
    
    @pytest.fixture
    def benchmark_framework(self, llm_config):
        """Create benchmark framework instance."""
        return BenchmarkFramework(llm_config=llm_config)
    
    def test_benchmark_framework_initialization(self, benchmark_framework):
        """Test benchmark framework initialization."""
        assert benchmark_framework is not None
        assert len(benchmark_framework.suites) == 0
        assert len(benchmark_framework.configurations) == 0
        assert len(benchmark_framework.results_history) == 0
    
    def test_add_suite(self, benchmark_framework, sample_benchmark_suite):
        """Test adding a benchmark suite."""
        benchmark_framework.add_suite(sample_benchmark_suite)
        
        assert len(benchmark_framework.suites) == 1
        assert sample_benchmark_suite.suite_id in benchmark_framework.suites
        assert benchmark_framework.suites[sample_benchmark_suite.suite_id] == sample_benchmark_suite
    
    def test_remove_suite(self, benchmark_framework, sample_benchmark_suite):
        """Test removing a benchmark suite."""
        # Add suite first
        benchmark_framework.add_suite(sample_benchmark_suite)
        assert len(benchmark_framework.suites) == 1
        
        # Remove suite
        result = benchmark_framework.remove_suite(sample_benchmark_suite.suite_id)
        assert result is True
        assert len(benchmark_framework.suites) == 0
        
        # Try to remove non-existent suite
        result = benchmark_framework.remove_suite("non-existent")
        assert result is False
    
    def test_get_suite(self, benchmark_framework, sample_benchmark_suite):
        """Test getting a benchmark suite."""
        # Test getting non-existent suite
        suite = benchmark_framework.get_suite("non-existent")
        assert suite is None
        
        # Add and get suite
        benchmark_framework.add_suite(sample_benchmark_suite)
        suite = benchmark_framework.get_suite(sample_benchmark_suite.suite_id)
        assert suite == sample_benchmark_suite
    
    def test_list_suites(self, benchmark_framework, sample_benchmark_suite):
        """Test listing all benchmark suites."""
        # Empty list initially
        suites = benchmark_framework.list_suites()
        assert len(suites) == 0
        
        # Add suite and list
        benchmark_framework.add_suite(sample_benchmark_suite)
        suites = benchmark_framework.list_suites()
        assert len(suites) == 1
        assert suites[0] == sample_benchmark_suite
    
    def test_add_configuration(self, benchmark_framework, sample_agent_config):
        """Test adding benchmark configuration."""
        config = BenchmarkConfiguration(
            suite_ids=["test-suite"],
            agent_configurations=[sample_agent_config],
            parallel_execution=True,
            max_concurrent_tests=3
        )
        
        benchmark_framework.add_configuration(config)
        assert len(benchmark_framework.configurations) == 1
        assert config.config_id in benchmark_framework.configurations
    
    @pytest.mark.asyncio
    async def test_run_benchmark_suite_not_found(self, benchmark_framework, sample_agent_config):
        """Test running benchmark suite that doesn't exist."""
        with pytest.raises(DataQAError, match="Benchmark suite not found"):
            await benchmark_framework.run_benchmark_suite(
                "non-existent", sample_agent_config
            )
    
    @pytest.mark.asyncio
    async def test_run_benchmark_suite_success(
        self, benchmark_framework, sample_benchmark_suite, sample_agent_config
    ):
        """Test successful benchmark suite execution."""
        # Add suite
        benchmark_framework.add_suite(sample_benchmark_suite)
        
        # Mock agent executor
        async def mock_executor(inputs, agent_config):
            return {"result": "success", "value": 42}
        
        # Run benchmark
        report = await benchmark_framework.run_benchmark_suite(
            sample_benchmark_suite.suite_id,
            sample_agent_config,
            mock_executor
        )
        
        # Verify report
        assert isinstance(report, BenchmarkReport)
        assert report.suite_id == sample_benchmark_suite.suite_id
        assert report.agent_configuration_id == sample_agent_config.agent_id
        assert report.total_test_cases == 1
        assert report.passed_test_cases >= 0
        assert report.failed_test_cases >= 0
        assert len(report.individual_results) == 1
        
        # Verify result is stored in history
        assert len(benchmark_framework.results_history) == 1
    
    @pytest.mark.asyncio
    async def test_run_benchmark_suite_with_failure(
        self, benchmark_framework, sample_benchmark_suite, sample_agent_config
    ):
        """Test benchmark suite execution with test failures."""
        # Add suite
        benchmark_framework.add_suite(sample_benchmark_suite)
        
        # Mock agent executor that fails
        async def mock_executor(inputs, agent_config):
            raise Exception("Agent execution failed")
        
        # Run benchmark
        report = await benchmark_framework.run_benchmark_suite(
            sample_benchmark_suite.suite_id,
            sample_agent_config,
            mock_executor
        )
        
        # Verify report shows failure
        assert report.total_test_cases == 1
        assert report.failed_test_cases == 1
        assert report.passed_test_cases == 0
        assert len(report.individual_results) == 1
        assert not report.individual_results[0].success
        assert report.individual_results[0].error_message is not None
    
    @pytest.mark.asyncio
    async def test_run_benchmark_configuration_not_found(self, benchmark_framework):
        """Test running benchmark configuration that doesn't exist."""
        with pytest.raises(DataQAError, match="Benchmark configuration not found"):
            await benchmark_framework.run_benchmark_configuration("non-existent")
    
    @pytest.mark.asyncio
    async def test_run_benchmark_configuration_success(
        self, benchmark_framework, sample_benchmark_suite, sample_agent_config
    ):
        """Test successful benchmark configuration execution."""
        # Add suite
        benchmark_framework.add_suite(sample_benchmark_suite)
        
        # Create configuration
        config = BenchmarkConfiguration(
            suite_ids=[sample_benchmark_suite.suite_id],
            agent_configurations=[sample_agent_config],
            generate_detailed_reports=False
        )
        benchmark_framework.add_configuration(config)
        
        # Mock agent executor
        async def mock_executor(inputs, agent_config):
            return {"result": "success", "value": 42}
        
        # Run benchmark configuration
        reports = await benchmark_framework.run_benchmark_configuration(
            config.config_id, mock_executor
        )
        
        # Verify reports
        assert len(reports) == 1
        assert isinstance(reports[0], BenchmarkReport)
        assert reports[0].suite_id == sample_benchmark_suite.suite_id
    
    def test_exact_match_evaluate(self, benchmark_framework, sample_ground_truth):
        """Test exact match evaluation method."""
        criterion = EvaluationCriterion(
            name="test",
            description="test",
            scoring_method="exact_match"
        )
        
        # Test exact match
        score = benchmark_framework._exact_match_evaluate(
            {"result": "success", "value": 42},
            sample_ground_truth,
            criterion
        )
        assert score == 1.0
        
        # Test acceptable variation match
        score = benchmark_framework._exact_match_evaluate(
            {"result": "success", "value": 41},
            sample_ground_truth,
            criterion
        )
        assert score == 1.0
        
        # Test no match
        score = benchmark_framework._exact_match_evaluate(
            {"result": "failure", "value": 0},
            sample_ground_truth,
            criterion
        )
        assert score == 0.0
    
    def test_similarity_evaluate(self, benchmark_framework):
        """Test similarity evaluation method."""
        ground_truth = GroundTruth(expected_output="hello world test")
        criterion = EvaluationCriterion(
            name="test",
            description="test",
            scoring_method="similarity"
        )
        
        # Test identical strings
        score = benchmark_framework._similarity_evaluate(
            "hello world test",
            ground_truth,
            criterion
        )
        assert score == 1.0
        
        # Test partial similarity
        score = benchmark_framework._similarity_evaluate(
            "hello world",
            ground_truth,
            criterion
        )
        assert score > 0.0 and score < 1.0
        
        # Test no similarity
        score = benchmark_framework._similarity_evaluate(
            "completely different",
            ground_truth,
            criterion
        )
        assert score >= 0.0 and score <= 1.0
    
    def test_check_quality_thresholds(self, benchmark_framework, sample_benchmark_suite):
        """Test quality threshold checking."""
        # Create report with metrics
        report = BenchmarkReport(
            suite_id=sample_benchmark_suite.suite_id,
            agent_configuration_id="test-agent",
            execution_start=datetime.utcnow(),
            execution_end=datetime.utcnow(),
            total_test_cases=10,
            passed_test_cases=7,
            failed_test_cases=3,
            average_execution_time=15.0,
            overall_score=0.7,
            performance_metrics={
                "success_rate": 0.7,
                "average_execution_time": 15.0
            }
        )
        
        # Check thresholds
        violations = benchmark_framework._check_quality_thresholds(
            report, sample_benchmark_suite.quality_thresholds
        )
        
        # Should have violation for success rate below 0.8
        assert len(violations) == 1
        assert "success_rate" in violations[0]
        assert "below minimum threshold" in violations[0]
    
    def test_generate_recommendations(self, benchmark_framework, sample_benchmark_suite):
        """Test recommendation generation."""
        # Create report with poor performance
        report = BenchmarkReport(
            suite_id=sample_benchmark_suite.suite_id,
            agent_configuration_id="test-agent",
            execution_start=datetime.utcnow(),
            execution_end=datetime.utcnow(),
            total_test_cases=10,
            passed_test_cases=6,
            failed_test_cases=4,
            average_execution_time=35.0,
            overall_score=0.6,
            performance_metrics={
                "success_rate": 0.6,
                "average_execution_time": 35.0
            },
            individual_results=[]
        )
        
        # Generate recommendations
        recommendations = benchmark_framework._generate_recommendations(
            report, sample_benchmark_suite
        )
        
        # Should have recommendations for poor performance
        assert len(recommendations) > 0
        assert any("success rate" in rec.lower() for rec in recommendations)
        assert any("execution time" in rec.lower() for rec in recommendations)
        assert any("quality score" in rec.lower() for rec in recommendations)
    
    def test_get_results_history(self, benchmark_framework):
        """Test getting results history."""
        # Initially empty
        history = benchmark_framework.get_results_history()
        assert len(history) == 0
        
        # Add some results
        report = BenchmarkReport(
            suite_id="test-suite",
            agent_configuration_id="test-agent",
            execution_start=datetime.utcnow(),
            execution_end=datetime.utcnow(),
            total_test_cases=1,
            passed_test_cases=1,
            failed_test_cases=0,
            average_execution_time=5.0,
            overall_score=0.9
        )
        benchmark_framework.results_history.append(report)
        
        # Get history
        history = benchmark_framework.get_results_history()
        assert len(history) == 1
        assert history[0] == report
    
    def test_get_performance_trends(self, benchmark_framework):
        """Test getting performance trends."""
        agent_id = "test-agent"
        
        # Add some historical reports
        for i in range(3):
            report = BenchmarkReport(
                suite_id="test-suite",
                agent_configuration_id=agent_id,
                execution_start=datetime.utcnow(),
                execution_end=datetime.utcnow(),
                total_test_cases=1,
                passed_test_cases=1,
                failed_test_cases=0,
                average_execution_time=5.0 + i,
                overall_score=0.9,
                performance_metrics={
                    "success_rate": 0.9 - (i * 0.1),
                    "average_execution_time": 5.0 + i
                }
            )
            benchmark_framework.results_history.append(report)
        
        # Get trends
        trends = benchmark_framework.get_performance_trends(agent_id, "success_rate")
        assert len(trends) == 3
        assert trends == [0.9, 0.8, 0.7]
        
        # Test non-existent metric
        trends = benchmark_framework.get_performance_trends(agent_id, "non_existent")
        assert len(trends) == 0


class TestBenchmarkModels:
    """Test cases for benchmark data models."""
    
    def test_test_case_creation(self):
        """Test TestCase model creation."""
        ground_truth = GroundTruth(expected_output="test result")
        
        test_case = TestCase(
            name="Test Case",
            description="A test case",
            inputs={"query": "test"},
            ground_truth=ground_truth
        )
        
        assert test_case.name == "Test Case"
        assert test_case.description == "A test case"
        assert test_case.category == "general"
        assert test_case.difficulty_level == "medium"
        assert test_case.timeout_seconds == 300
        assert test_case.ground_truth == ground_truth
    
    def test_benchmark_suite_creation(self):
        """Test BenchmarkSuite model creation."""
        suite = BenchmarkSuite(
            name="Test Suite",
            description="A test suite",
            version="2.0.0"
        )
        
        assert suite.name == "Test Suite"
        assert suite.description == "A test suite"
        assert suite.version == "2.0.0"
        assert len(suite.test_cases) == 0
        assert len(suite.evaluation_criteria) == 0
        assert isinstance(suite.created_at, datetime)
        assert isinstance(suite.updated_at, datetime)
    
    def test_evaluation_criterion_creation(self):
        """Test EvaluationCriterion model creation."""
        criterion = EvaluationCriterion(
            name="Accuracy",
            description="Measures accuracy",
            weight=0.8,
            scoring_method="llm_judge",
            parameters={"threshold": 0.9}
        )
        
        assert criterion.name == "Accuracy"
        assert criterion.description == "Measures accuracy"
        assert criterion.weight == 0.8
        assert criterion.scoring_method == "llm_judge"
        assert criterion.parameters["threshold"] == 0.9
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult model creation."""
        result = BenchmarkResult(
            test_case_id="test-case-1",
            agent_configuration_id="agent-1",
            execution_time_seconds=5.5,
            success=True,
            agent_response={"answer": "42"},
            evaluation_scores={"accuracy": 0.95}
        )
        
        assert result.test_case_id == "test-case-1"
        assert result.agent_configuration_id == "agent-1"
        assert result.execution_time_seconds == 5.5
        assert result.success is True
        assert result.agent_response == {"answer": "42"}
        assert result.evaluation_scores["accuracy"] == 0.95
        assert isinstance(result.timestamp, datetime)
    
    def test_benchmark_report_creation(self):
        """Test BenchmarkReport model creation."""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()
        
        report = BenchmarkReport(
            suite_id="suite-1",
            agent_configuration_id="agent-1",
            execution_start=start_time,
            execution_end=end_time,
            total_test_cases=10,
            passed_test_cases=8,
            failed_test_cases=2,
            average_execution_time=12.5,
            overall_score=0.85
        )
        
        assert report.suite_id == "suite-1"
        assert report.agent_configuration_id == "agent-1"
        assert report.execution_start == start_time
        assert report.execution_end == end_time
        assert report.total_test_cases == 10
        assert report.passed_test_cases == 8
        assert report.failed_test_cases == 2
        assert report.average_execution_time == 12.5
        assert report.overall_score == 0.85
    
    def test_benchmark_configuration_creation(self):
        """Test BenchmarkConfiguration model creation."""
        agent_config = AgentConfiguration(
            name="Test Agent",
            role=AgentRole.WORKER
        )
        
        config = BenchmarkConfiguration(
            suite_ids=["suite-1", "suite-2"],
            agent_configurations=[agent_config],
            parallel_execution=True,
            max_concurrent_tests=3,
            timeout_seconds=300,
            retry_failed_tests=True
        )
        
        assert config.suite_ids == ["suite-1", "suite-2"]
        assert len(config.agent_configurations) == 1
        assert config.parallel_execution is True
        assert config.max_concurrent_tests == 3
        assert config.timeout_seconds == 300
        assert config.retry_failed_tests is True