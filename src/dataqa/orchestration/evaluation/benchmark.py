"""
Comprehensive benchmarking framework for multi-agent system evaluation.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import uuid4
from pathlib import Path
import json

from pydantic import BaseModel, Field, validator

from ...config.models import LLMConfig
from ...primitives.llm import LLMInterface, create_llm_interface
from ...logging_config import get_primitive_logger
from ...exceptions import DataQAError
from ..models import AgentConfiguration, ExecutionSession, ExecutionMetrics


class GroundTruth(BaseModel):
    """Ground truth data for evaluation."""
    truth_id: str = Field(default_factory=lambda: str(uuid4()))
    expected_output: Any
    acceptable_variations: List[Any] = Field(default_factory=list)
    evaluation_criteria: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    """Individual test case for benchmarking."""
    test_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    category: str = "general"
    difficulty_level: str = "medium"  # easy, medium, hard
    inputs: Dict[str, Any] = Field(default_factory=dict)
    ground_truth: GroundTruth
    timeout_seconds: int = 300
    retry_count: int = 0
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationCriterion(BaseModel):
    """Evaluation criterion for scoring responses."""
    criterion_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    weight: float = 1.0
    scoring_method: str = "llm_judge"  # llm_judge, exact_match, similarity, custom
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceBaseline(BaseModel):
    """Performance baseline for comparison."""
    baseline_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    agent_configuration: Optional[str] = None
    established_date: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityThreshold(BaseModel):
    """Quality threshold for pass/fail determination."""
    threshold_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_name: str
    minimum_value: float
    maximum_value: Optional[float] = None
    severity: str = "error"  # error, warning, info
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuite(BaseModel):
    """Collection of test cases for benchmarking."""
    suite_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    version: str = "1.0.0"
    test_cases: List[TestCase] = Field(default_factory=list)
    evaluation_criteria: List[EvaluationCriterion] = Field(default_factory=list)
    performance_baselines: Dict[str, PerformanceBaseline] = Field(default_factory=dict)
    quality_thresholds: Dict[str, QualityThreshold] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Result of running a single test case."""
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    test_case_id: str
    agent_configuration_id: str
    execution_time_seconds: float
    success: bool
    agent_response: Any
    evaluation_scores: Dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkReport(BaseModel):
    """Comprehensive report of benchmark execution."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    suite_id: str
    agent_configuration_id: str
    execution_start: datetime
    execution_end: datetime
    total_test_cases: int
    passed_test_cases: int
    failed_test_cases: int
    average_execution_time: float
    overall_score: float
    individual_results: List[BenchmarkResult] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    quality_violations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfiguration(BaseModel):
    """Configuration for benchmark execution."""
    config_id: str = Field(default_factory=lambda: str(uuid4()))
    suite_ids: List[str]
    agent_configurations: List[AgentConfiguration]
    llm_judge_config: Optional[LLMConfig] = None
    parallel_execution: bool = False
    max_concurrent_tests: int = 5
    timeout_seconds: int = 600
    retry_failed_tests: bool = True
    generate_detailed_reports: bool = True
    output_directory: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkFramework:
    """
    Comprehensive framework for managing and executing benchmark suites.
    
    Supports configurable test cases, ground truth data sources, LLM judge evaluation,
    and automated benchmark execution with business rules integration.
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize the benchmark framework.
        
        Args:
            llm_config: Optional LLM configuration for judge evaluation
        """
        self.logger = get_primitive_logger("benchmark", "framework")
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.configurations: Dict[str, BenchmarkConfiguration] = {}
        self.results_history: List[BenchmarkReport] = []
        
        # Initialize LLM judge if configuration provided
        self.llm_judge: Optional[LLMInterface] = None
        if llm_config:
            try:
                self.llm_judge = create_llm_interface(llm_config)
                self.logger.info("LLM judge initialized for evaluation")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM judge: {e}")
    
    def add_suite(self, suite: BenchmarkSuite) -> None:
        """Add a benchmark suite to the framework.
        
        Args:
            suite: Benchmark suite to add
        """
        self.suites[suite.suite_id] = suite
        self.logger.info(f"Added benchmark suite: {suite.name} ({suite.suite_id})")
    
    def remove_suite(self, suite_id: str) -> bool:
        """Remove a benchmark suite.
        
        Args:
            suite_id: ID of suite to remove
            
        Returns:
            True if suite was removed, False if not found
        """
        if suite_id in self.suites:
            del self.suites[suite_id]
            self.logger.info(f"Removed benchmark suite: {suite_id}")
            return True
        return False
    
    def get_suite(self, suite_id: str) -> Optional[BenchmarkSuite]:
        """Get a benchmark suite by ID.
        
        Args:
            suite_id: ID of suite to retrieve
            
        Returns:
            Benchmark suite or None if not found
        """
        return self.suites.get(suite_id)
    
    def list_suites(self) -> List[BenchmarkSuite]:
        """List all available benchmark suites.
        
        Returns:
            List of all benchmark suites
        """
        return list(self.suites.values())
    
    def add_configuration(self, config: BenchmarkConfiguration) -> None:
        """Add a benchmark configuration.
        
        Args:
            config: Benchmark configuration to add
        """
        self.configurations[config.config_id] = config
        self.logger.info(f"Added benchmark configuration: {config.config_id}")
    
    async def run_benchmark_suite(
        self,
        suite_id: str,
        agent_config: AgentConfiguration,
        agent_executor: Optional[Callable] = None
    ) -> BenchmarkReport:
        """Run a complete benchmark suite against an agent configuration.
        
        Args:
            suite_id: ID of benchmark suite to run
            agent_config: Agent configuration to test
            agent_executor: Optional custom agent executor function
            
        Returns:
            Comprehensive benchmark report
            
        Raises:
            DataQAError: If suite not found or execution fails
        """
        suite = self.suites.get(suite_id)
        if not suite:
            raise DataQAError(f"Benchmark suite not found: {suite_id}")
        
        self.logger.info(f"Starting benchmark suite: {suite.name} with agent: {agent_config.name}")
        
        start_time = datetime.utcnow()
        results: List[BenchmarkResult] = []
        
        # Execute each test case
        for test_case in suite.test_cases:
            try:
                result = await self._run_test_case(
                    test_case, agent_config, suite.evaluation_criteria, agent_executor
                )
                results.append(result)
                self.logger.debug(f"Completed test case: {test_case.name}")
            except Exception as e:
                self.logger.error(f"Failed test case {test_case.name}: {e}")
                # Create failed result
                failed_result = BenchmarkResult(
                    test_case_id=test_case.test_id,
                    agent_configuration_id=agent_config.agent_id,
                    execution_time_seconds=0.0,
                    success=False,
                    agent_response=None,
                    error_message=str(e)
                )
                results.append(failed_result)
        
        end_time = datetime.utcnow()
        
        # Generate comprehensive report
        report = self._generate_report(
            suite, agent_config, results, start_time, end_time
        )
        
        # Check quality thresholds and add violations
        violations = self._check_quality_thresholds(report, suite.quality_thresholds)
        report.quality_violations = violations
        
        # Generate recommendations
        recommendations = self._generate_recommendations(report, suite)
        report.recommendations = recommendations
        
        self.results_history.append(report)
        self.logger.info(f"Completed benchmark suite: {suite.name}")
        
        return report
    
    async def run_benchmark_configuration(
        self,
        config_id: str,
        agent_executor: Optional[Callable] = None
    ) -> List[BenchmarkReport]:
        """Run a benchmark configuration across multiple suites and agents.
        
        Args:
            config_id: ID of benchmark configuration to run
            agent_executor: Optional custom agent executor function
            
        Returns:
            List of benchmark reports for each suite/agent combination
            
        Raises:
            DataQAError: If configuration not found
        """
        config = self.configurations.get(config_id)
        if not config:
            raise DataQAError(f"Benchmark configuration not found: {config_id}")
        
        self.logger.info(f"Starting benchmark configuration: {config_id}")
        
        reports: List[BenchmarkReport] = []
        
        # Run each suite against each agent configuration
        for suite_id in config.suite_ids:
            for agent_config in config.agent_configurations:
                try:
                    report = await self.run_benchmark_suite(
                        suite_id, agent_config, agent_executor
                    )
                    reports.append(report)
                except Exception as e:
                    self.logger.error(f"Failed benchmark {suite_id} with {agent_config.name}: {e}")
        
        # Save reports if output directory specified
        if config.output_directory and config.generate_detailed_reports:
            await self._save_reports(reports, config.output_directory)
        
        self.logger.info(f"Completed benchmark configuration: {config_id}")
        return reports
    
    async def _run_test_case(
        self,
        test_case: TestCase,
        agent_config: AgentConfiguration,
        evaluation_criteria: List[EvaluationCriterion],
        agent_executor: Optional[Callable] = None
    ) -> BenchmarkResult:
        """Run a single test case.
        
        Args:
            test_case: Test case to run
            agent_config: Agent configuration to test
            evaluation_criteria: Criteria for evaluation
            agent_executor: Optional custom agent executor
            
        Returns:
            Benchmark result for the test case
        """
        start_time = datetime.utcnow()
        
        try:
            # Execute agent with test case inputs
            if agent_executor:
                agent_response = await agent_executor(test_case.inputs, agent_config)
            else:
                # Default mock execution for testing
                agent_response = {"mock_response": "test_output"}
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Evaluate response against ground truth
            evaluation_scores = await self._evaluate_response(
                agent_response, test_case.ground_truth, evaluation_criteria
            )
            
            # Determine success based on evaluation scores
            success = all(score >= 0.5 for score in evaluation_scores.values())
            
            return BenchmarkResult(
                test_case_id=test_case.test_id,
                agent_configuration_id=agent_config.agent_id,
                execution_time_seconds=execution_time,
                success=success,
                agent_response=agent_response,
                evaluation_scores=evaluation_scores
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return BenchmarkResult(
                test_case_id=test_case.test_id,
                agent_configuration_id=agent_config.agent_id,
                execution_time_seconds=execution_time,
                success=False,
                agent_response=None,
                error_message=str(e)
            )
    
    async def _evaluate_response(
        self,
        agent_response: Any,
        ground_truth: GroundTruth,
        evaluation_criteria: List[EvaluationCriterion]
    ) -> Dict[str, float]:
        """Evaluate agent response against ground truth using specified criteria.
        
        Args:
            agent_response: Response from agent
            ground_truth: Expected ground truth
            evaluation_criteria: Evaluation criteria to apply
            
        Returns:
            Dictionary of evaluation scores by criterion
        """
        scores: Dict[str, float] = {}
        
        for criterion in evaluation_criteria:
            try:
                if criterion.scoring_method == "llm_judge" and self.llm_judge:
                    score = await self._llm_judge_evaluate(
                        agent_response, ground_truth, criterion
                    )
                elif criterion.scoring_method == "exact_match":
                    score = self._exact_match_evaluate(
                        agent_response, ground_truth, criterion
                    )
                elif criterion.scoring_method == "similarity":
                    score = self._similarity_evaluate(
                        agent_response, ground_truth, criterion
                    )
                else:
                    # Default scoring
                    score = 0.5
                
                scores[criterion.name] = score
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate criterion {criterion.name}: {e}")
                scores[criterion.name] = 0.0
        
        return scores
    
    async def _llm_judge_evaluate(
        self,
        agent_response: Any,
        ground_truth: GroundTruth,
        criterion: EvaluationCriterion
    ) -> float:
        """Evaluate using LLM judge.
        
        Args:
            agent_response: Response from agent
            ground_truth: Expected ground truth
            criterion: Evaluation criterion
            
        Returns:
            Evaluation score between 0.0 and 1.0
        """
        if not self.llm_judge:
            return 0.5
        
        # Create evaluation prompt
        prompt = f"""
        Evaluate the following agent response against the ground truth based on the criterion: {criterion.name}
        
        Criterion Description: {criterion.description}
        
        Agent Response:
        {json.dumps(agent_response, indent=2, default=str)}
        
        Ground Truth:
        {json.dumps(ground_truth.expected_output, indent=2, default=str)}
        
        Provide a score between 0.0 and 1.0, where:
        - 1.0 = Perfect match/excellent quality
        - 0.8-0.9 = Very good with minor issues
        - 0.6-0.7 = Good with some issues
        - 0.4-0.5 = Acceptable but needs improvement
        - 0.2-0.3 = Poor quality
        - 0.0-0.1 = Completely incorrect/unusable
        
        Return only the numeric score.
        """
        
        try:
            response = await self.llm_judge.format_response(
                {"evaluation_prompt": prompt}, 
                "Evaluate the response quality"
            )
            
            # Extract numeric score from response
            score_str = response.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"LLM judge evaluation failed: {e}")
            return 0.5
    
    def _exact_match_evaluate(
        self,
        agent_response: Any,
        ground_truth: GroundTruth,
        criterion: EvaluationCriterion
    ) -> float:
        """Evaluate using exact match comparison.
        
        Args:
            agent_response: Response from agent
            ground_truth: Expected ground truth
            criterion: Evaluation criterion
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if agent_response == ground_truth.expected_output:
            return 1.0
        
        # Check acceptable variations
        for variation in ground_truth.acceptable_variations:
            if agent_response == variation:
                return 1.0
        
        return 0.0
    
    def _similarity_evaluate(
        self,
        agent_response: Any,
        ground_truth: GroundTruth,
        criterion: EvaluationCriterion
    ) -> float:
        """Evaluate using similarity comparison.
        
        Args:
            agent_response: Response from agent
            ground_truth: Expected ground truth
            criterion: Evaluation criterion
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple string similarity for now
        if isinstance(agent_response, str) and isinstance(ground_truth.expected_output, str):
            response_words = set(agent_response.lower().split())
            truth_words = set(ground_truth.expected_output.lower().split())
            
            if not truth_words:
                return 1.0 if not response_words else 0.0
            
            intersection = response_words.intersection(truth_words)
            union = response_words.union(truth_words)
            
            return len(intersection) / len(union) if union else 1.0
        
        # Default similarity for non-string types
        return 1.0 if agent_response == ground_truth.expected_output else 0.0
    
    def _generate_report(
        self,
        suite: BenchmarkSuite,
        agent_config: AgentConfiguration,
        results: List[BenchmarkResult],
        start_time: datetime,
        end_time: datetime
    ) -> BenchmarkReport:
        """Generate comprehensive benchmark report.
        
        Args:
            suite: Benchmark suite that was run
            agent_config: Agent configuration tested
            results: Individual test results
            start_time: Execution start time
            end_time: Execution end time
            
        Returns:
            Comprehensive benchmark report
        """
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        avg_execution_time = (
            sum(r.execution_time_seconds for r in results) / total_tests
            if total_tests > 0 else 0.0
        )
        
        # Calculate overall score as weighted average of evaluation scores
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            for criterion_name, score in result.evaluation_scores.items():
                # Find criterion weight
                weight = 1.0
                for criterion in suite.evaluation_criteria:
                    if criterion.name == criterion_name:
                        weight = criterion.weight
                        break
                
                total_score += score * weight
                total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate performance metrics
        performance_metrics = {
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "average_execution_time": avg_execution_time,
            "throughput": total_tests / (end_time - start_time).total_seconds(),
            "error_rate": failed_tests / total_tests if total_tests > 0 else 0.0
        }
        
        return BenchmarkReport(
            suite_id=suite.suite_id,
            agent_configuration_id=agent_config.agent_id,
            execution_start=start_time,
            execution_end=end_time,
            total_test_cases=total_tests,
            passed_test_cases=passed_tests,
            failed_test_cases=failed_tests,
            average_execution_time=avg_execution_time,
            overall_score=overall_score,
            individual_results=results,
            performance_metrics=performance_metrics
        )
    
    def _check_quality_thresholds(
        self,
        report: BenchmarkReport,
        thresholds: Dict[str, QualityThreshold]
    ) -> List[str]:
        """Check quality thresholds and return violations.
        
        Args:
            report: Benchmark report to check
            thresholds: Quality thresholds to check against
            
        Returns:
            List of quality violation messages
        """
        violations: List[str] = []
        
        for threshold_name, threshold in thresholds.items():
            metric_value = report.performance_metrics.get(threshold.metric_name)
            
            if metric_value is None:
                continue
            
            if metric_value < threshold.minimum_value:
                violations.append(
                    f"{threshold.metric_name} ({metric_value:.3f}) below minimum threshold "
                    f"({threshold.minimum_value:.3f}): {threshold.description}"
                )
            
            if threshold.maximum_value and metric_value > threshold.maximum_value:
                violations.append(
                    f"{threshold.metric_name} ({metric_value:.3f}) above maximum threshold "
                    f"({threshold.maximum_value:.3f}): {threshold.description}"
                )
        
        return violations
    
    def _generate_recommendations(
        self,
        report: BenchmarkReport,
        suite: BenchmarkSuite
    ) -> List[str]:
        """Generate optimization recommendations based on benchmark results.
        
        Args:
            report: Benchmark report to analyze
            suite: Benchmark suite that was run
            
        Returns:
            List of optimization recommendations
        """
        recommendations: List[str] = []
        
        # Performance-based recommendations
        if report.performance_metrics["success_rate"] < 0.8:
            recommendations.append(
                "Success rate is below 80%. Consider reviewing agent logic and error handling."
            )
        
        if report.performance_metrics["average_execution_time"] > 30.0:
            recommendations.append(
                "Average execution time is high. Consider optimizing agent performance or increasing timeout."
            )
        
        if report.overall_score < 0.7:
            recommendations.append(
                "Overall quality score is below 70%. Review evaluation criteria and agent responses."
            )
        
        # Failure pattern analysis
        failed_categories = {}
        for result in report.individual_results:
            if not result.success:
                test_case = next(
                    (tc for tc in suite.test_cases if tc.test_id == result.test_case_id),
                    None
                )
                if test_case:
                    category = test_case.category
                    failed_categories[category] = failed_categories.get(category, 0) + 1
        
        if failed_categories:
            worst_category = max(failed_categories.items(), key=lambda x: x[1])
            recommendations.append(
                f"High failure rate in '{worst_category[0]}' category ({worst_category[1]} failures). "
                f"Focus improvement efforts on this area."
            )
        
        return recommendations
    
    async def _save_reports(
        self,
        reports: List[BenchmarkReport],
        output_directory: str
    ) -> None:
        """Save benchmark reports to files.
        
        Args:
            reports: Reports to save
            output_directory: Directory to save reports in
        """
        try:
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for report in reports:
                filename = f"benchmark_report_{report.report_id}.json"
                filepath = output_path / filename
                
                with open(filepath, 'w') as f:
                    json.dump(report.dict(), f, indent=2, default=str)
                
                self.logger.info(f"Saved benchmark report: {filepath}")
                
        except Exception as e:
            self.logger.error(f"Failed to save reports: {e}")
    
    def get_results_history(self) -> List[BenchmarkReport]:
        """Get historical benchmark results.
        
        Returns:
            List of all benchmark reports
        """
        return self.results_history.copy()
    
    def get_performance_trends(self, agent_id: str, metric_name: str) -> List[float]:
        """Get performance trends for a specific agent and metric.
        
        Args:
            agent_id: Agent configuration ID
            metric_name: Name of metric to track
            
        Returns:
            List of metric values over time
        """
        trends = []
        for report in self.results_history:
            if report.agent_configuration_id == agent_id:
                value = report.performance_metrics.get(metric_name)
                if value is not None:
                    trends.append(value)
        return trends