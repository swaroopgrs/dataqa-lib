"""
Tests for the advanced performance analytics system.
"""

import pytest
import statistics
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.orchestration.evaluation.analytics import (
    PerformanceAnalytics,
    PerformanceMetrics,
    MetricsSnapshot,
    TrendAnalysis,
    Optimization,
    PerformanceData,
    MetricsCollector,
    TrendAnalyzer,
    OptimizationRecommender
)
from src.dataqa.orchestration.models import (
    ExecutionSession,
    ExecutionState,
    ExecutionMetrics,
    MultiAgentWorkflow
)
from src.dataqa.exceptions import DataQAError


class TestPerformanceAnalytics:
    """Test cases for PerformanceAnalytics."""
    
    @pytest.fixture
    def sample_execution_metrics(self):
        """Create sample execution metrics."""
        return ExecutionMetrics(
            total_steps=10,
            completed_steps=8,
            failed_steps=2,
            total_execution_time_seconds=25.5,
            average_step_time_seconds=2.55,
            resource_utilization={
                "cpu_percent": 45.0,
                "memory_mb": 512.0,
                "disk_io_mb": 10.5
            },
            quality_scores={
                "accuracy": 0.85,
                "completeness": 0.9,
                "relevance": 0.8
            }
        )
    
    @pytest.fixture
    def sample_execution_session(self, sample_execution_metrics):
        """Create sample execution session."""
        execution_state = ExecutionState(
            execution_metrics=sample_execution_metrics,
            status="completed"
        )
        
        return ExecutionSession(
            workflow_id="test-workflow",
            execution_state=execution_state
        )
    
    @pytest.fixture
    def performance_analytics(self):
        """Create PerformanceAnalytics instance."""
        return PerformanceAnalytics()
    
    def test_analytics_initialization(self, performance_analytics):
        """Test performance analytics initialization."""
        assert performance_analytics is not None
        assert len(performance_analytics.metrics_history) == 0
        assert len(performance_analytics.snapshots_history) == 0
        assert len(performance_analytics.trend_analyses) == 0
        assert len(performance_analytics.optimizations) == 0
        
        # Should have default configurations
        assert len(performance_analytics.metrics_collectors) > 0
        assert len(performance_analytics.trend_analyzers) > 0
        assert len(performance_analytics.optimization_recommenders) > 0
    
    def test_add_metrics_collector(self, performance_analytics):
        """Test adding metrics collector."""
        collector = MetricsCollector(
            name="Test Collector",
            description="A test metrics collector",
            metric_type="custom",
            collection_interval_seconds=60,
            query_pattern="test.*"
        )
        
        initial_count = len(performance_analytics.metrics_collectors)
        performance_analytics.add_metrics_collector(collector)
        
        assert len(performance_analytics.metrics_collectors) == initial_count + 1
        assert collector in performance_analytics.metrics_collectors
    
    def test_add_trend_analyzer(self, performance_analytics):
        """Test adding trend analyzer."""
        analyzer = TrendAnalyzer(
            name="Test Analyzer",
            description="A test trend analyzer",
            metric_names=["test_metric"],
            analysis_window_hours=12,
            trend_detection_method="moving_average"
        )
        
        initial_count = len(performance_analytics.trend_analyzers)
        performance_analytics.add_trend_analyzer(analyzer)
        
        assert len(performance_analytics.trend_analyzers) == initial_count + 1
        assert analyzer in performance_analytics.trend_analyzers
    
    def test_add_optimization_recommender(self, performance_analytics):
        """Test adding optimization recommender."""
        recommender = OptimizationRecommender(
            name="Test Recommender",
            description="A test optimization recommender",
            target_metrics=["test_metric"],
            optimization_strategy="performance"
        )
        
        initial_count = len(performance_analytics.optimization_recommenders)
        performance_analytics.add_optimization_recommender(recommender)
        
        assert len(performance_analytics.optimization_recommenders) == initial_count + 1
        assert recommender in performance_analytics.optimization_recommenders
    
    @pytest.mark.asyncio
    async def test_collect_metrics_success(self, performance_analytics, sample_execution_session):
        """Test successful metrics collection."""
        snapshot = await performance_analytics.collect_metrics(sample_execution_session)
        
        # Verify snapshot
        assert isinstance(snapshot, MetricsSnapshot)
        assert len(snapshot.metrics) > 0
        assert len(snapshot.aggregated_metrics) > 0
        
        # Verify base metrics
        base_metrics = snapshot.metrics[0]
        assert base_metrics.session_id == sample_execution_session.session_id
        assert base_metrics.execution_time_seconds == 25.5
        assert base_metrics.success_rate == 0.8  # 8/10
        assert base_metrics.error_rate == 0.2   # 2/10
        assert base_metrics.resource_utilization["cpu_percent"] == 45.0
        assert base_metrics.quality_scores["accuracy"] == 0.85
        
        # Verify aggregated metrics
        assert "avg_execution_time" in snapshot.aggregated_metrics
        assert "avg_success_rate" in snapshot.aggregated_metrics
        
        # Verify storage in history
        assert len(performance_analytics.metrics_history) > 0
        assert len(performance_analytics.snapshots_history) == 1
        assert performance_analytics.snapshots_history[0] == snapshot
    
    @pytest.mark.asyncio
    async def test_collect_metrics_with_failed_collector(self, performance_analytics, sample_execution_session):
        """Test metrics collection with a failing collector."""
        # Add a collector that will fail
        failing_collector = MetricsCollector(
            name="Failing Collector",
            description="This collector will fail",
            metric_type="business",
            enabled=True
        )
        performance_analytics.add_metrics_collector(failing_collector)
        
        # Mock the _apply_collector method to raise an exception
        original_apply = performance_analytics._apply_collector
        async def mock_apply_collector(session, collector):
            if collector.name == "Failing Collector":
                raise Exception("Collector failed")
            return await original_apply(session, collector)
        
        performance_analytics._apply_collector = mock_apply_collector
        
        # Should still succeed despite failing collector
        snapshot = await performance_analytics.collect_metrics(sample_execution_session)
        assert isinstance(snapshot, MetricsSnapshot)
        assert len(snapshot.metrics) > 0
    
    def test_extract_base_metrics(self, performance_analytics, sample_execution_session):
        """Test extracting base metrics from execution session."""
        metrics = performance_analytics._extract_base_metrics(sample_execution_session)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.session_id == sample_execution_session.session_id
        assert metrics.execution_time_seconds == 25.5
        assert metrics.success_rate == 0.8
        assert metrics.error_rate == 0.2
        assert metrics.throughput > 0  # Should be calculated
        assert len(metrics.resource_utilization) > 0
        assert len(metrics.quality_scores) > 0
    
    def test_aggregate_metrics(self, performance_analytics):
        """Test metrics aggregation."""
        metrics = [
            PerformanceMetrics(
                session_id="session-1",
                agent_configuration_id="agent-1",
                execution_time_seconds=10.0,
                success_rate=0.9,
                throughput=5.0,
                error_rate=0.1
            ),
            PerformanceMetrics(
                session_id="session-2",
                agent_configuration_id="agent-1",
                execution_time_seconds=20.0,
                success_rate=0.8,
                throughput=3.0,
                error_rate=0.2
            ),
            PerformanceMetrics(
                session_id="session-3",
                agent_configuration_id="agent-1",
                execution_time_seconds=0.0,  # Should be excluded from time calculations
                success_rate=0.7,
                throughput=0.0,  # Should be excluded from throughput calculations
                error_rate=0.3
            )
        ]
        
        aggregated = performance_analytics._aggregate_metrics(metrics)
        
        # Check aggregated values
        assert aggregated["avg_execution_time"] == 15.0  # (10 + 20) / 2
        assert aggregated["max_execution_time"] == 20.0
        assert aggregated["min_execution_time"] == 10.0
        assert aggregated["avg_success_rate"] == 0.8  # (0.9 + 0.8 + 0.7) / 3
        assert aggregated["avg_throughput"] == 4.0  # (5 + 3) / 2
        assert aggregated["avg_error_rate"] == 0.2  # (0.1 + 0.2 + 0.3) / 3
    
    def test_aggregate_metrics_empty_list(self, performance_analytics):
        """Test metrics aggregation with empty list."""
        aggregated = performance_analytics._aggregate_metrics([])
        assert aggregated == {}
    
    def test_update_real_time_metrics(self, performance_analytics):
        """Test updating real-time metrics storage."""
        metrics = [
            PerformanceMetrics(
                session_id="session-1",
                agent_configuration_id="agent-1",
                execution_time_seconds=10.0,
                success_rate=0.9,
                throughput=5.0
            )
        ]
        
        initial_count = len(performance_analytics.real_time_metrics["execution_time"])
        performance_analytics._update_real_time_metrics(metrics)
        
        # Verify metrics were added
        assert len(performance_analytics.real_time_metrics["execution_time"]) == initial_count + 1
        assert len(performance_analytics.real_time_metrics["success_rate"]) == initial_count + 1
        assert len(performance_analytics.real_time_metrics["throughput"]) == initial_count + 1
        
        # Verify values
        _, exec_time = performance_analytics.real_time_metrics["execution_time"][-1]
        _, success_rate = performance_analytics.real_time_metrics["success_rate"][-1]
        _, throughput = performance_analytics.real_time_metrics["throughput"][-1]
        
        assert exec_time == 10.0
        assert success_rate == 0.9
        assert throughput == 5.0
    
    @pytest.mark.asyncio
    async def test_analyze_trends_success(self, performance_analytics):
        """Test successful trend analysis."""
        # Create historical data
        snapshots = []
        for i in range(5):
            snapshot = MetricsSnapshot(
                aggregated_metrics={
                    "avg_execution_time_seconds": 10.0 + i,
                    "avg_success_rate": 0.9 - (i * 0.05),
                    "avg_throughput": 5.0 + i
                }
            )
            snapshots.append(snapshot)
        
        analysis = await performance_analytics.analyze_trends(snapshots)
        
        assert isinstance(analysis, TrendAnalysis)
        assert analysis.analyzer_name is not None
        assert analysis.metric_name is not None
        assert analysis.trend_direction in ["increasing", "decreasing", "stable", "insufficient_data"]
        assert 0.0 <= analysis.trend_strength <= 1.0
        assert 0.0 <= analysis.confidence_level <= 1.0
        assert len(analysis.key_insights) > 0
        
        # Verify it's stored in history
        assert len(performance_analytics.trend_analyses) == 1
        assert performance_analytics.trend_analyses[0] == analysis
    
    @pytest.mark.asyncio
    async def test_analyze_trends_insufficient_data(self, performance_analytics):
        """Test trend analysis with insufficient data."""
        # Single snapshot
        snapshots = [MetricsSnapshot(aggregated_metrics={"avg_execution_time_seconds": 10.0})]
        
        analysis = await performance_analytics.analyze_trends(snapshots)
        
        assert analysis.trend_direction == "insufficient_data"
        assert analysis.trend_strength == 0.0
        assert analysis.confidence_level == 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_trends_no_analyzers(self, performance_analytics):
        """Test trend analysis when no analyzers are enabled."""
        # Disable all analyzers
        for analyzer in performance_analytics.trend_analyzers:
            analyzer.enabled = False
        
        snapshots = [MetricsSnapshot(aggregated_metrics={"test": 1.0})]
        analysis = await performance_analytics.analyze_trends(snapshots)
        
        # Should return default analysis
        assert analysis.trend_direction == "unknown"
        assert analysis.confidence_level == 0.0
    
    def test_prepare_trend_data(self, performance_analytics):
        """Test preparing data for trend analysis."""
        analyzer = TrendAnalyzer(
            name="Test Analyzer",
            description="Test",
            metric_names=["execution_time_seconds", "success_rate"]
        )
        
        snapshots = [
            MetricsSnapshot(aggregated_metrics={
                "avg_execution_time_seconds": 10.0,
                "avg_success_rate": 0.9
            }),
            MetricsSnapshot(aggregated_metrics={
                "avg_execution_time_seconds": 12.0,
                "avg_success_rate": 0.8
            }),
            MetricsSnapshot(aggregated_metrics={
                "avg_execution_time_seconds": 15.0,
                "avg_success_rate": 0.7
            })
        ]
        
        trend_data = performance_analytics._prepare_trend_data(snapshots, analyzer)
        
        assert "execution_time_seconds" in trend_data
        assert "success_rate" in trend_data
        assert trend_data["execution_time_seconds"] == [10.0, 12.0, 15.0]
        assert trend_data["success_rate"] == [0.9, 0.8, 0.7]
    
    @pytest.mark.asyncio
    async def test_perform_trend_analysis_increasing(self, performance_analytics):
        """Test trend analysis detecting increasing trend."""
        analyzer = TrendAnalyzer(
            name="Test Analyzer",
            description="Test",
            metric_names=["execution_time_seconds"],
            analysis_window_hours=24
        )
        
        # Increasing trend data
        trend_data = {"execution_time_seconds": [10.0, 15.0, 20.0, 25.0]}
        
        analysis = await performance_analytics._perform_trend_analysis(trend_data, analyzer)
        
        assert analysis.trend_direction == "increasing"
        assert analysis.trend_strength > 0
        assert "increasing" in analysis.key_insights[0].lower()
    
    @pytest.mark.asyncio
    async def test_perform_trend_analysis_decreasing(self, performance_analytics):
        """Test trend analysis detecting decreasing trend."""
        analyzer = TrendAnalyzer(
            name="Test Analyzer",
            description="Test",
            metric_names=["success_rate"],
            analysis_window_hours=24
        )
        
        # Decreasing trend data
        trend_data = {"success_rate": [0.9, 0.8, 0.7, 0.6]}
        
        analysis = await performance_analytics._perform_trend_analysis(trend_data, analyzer)
        
        assert analysis.trend_direction == "decreasing"
        assert analysis.trend_strength > 0
        assert "declining" in analysis.key_insights[0].lower()
    
    @pytest.mark.asyncio
    async def test_perform_trend_analysis_stable(self, performance_analytics):
        """Test trend analysis detecting stable trend."""
        analyzer = TrendAnalyzer(
            name="Test Analyzer",
            description="Test",
            metric_names=["throughput"],
            analysis_window_hours=24
        )
        
        # Stable trend data
        trend_data = {"throughput": [5.0, 5.1, 4.9, 5.0]}
        
        analysis = await performance_analytics._perform_trend_analysis(trend_data, analyzer)
        
        assert analysis.trend_direction == "stable"
        assert "stable" in analysis.key_insights[0].lower()
    
    @pytest.mark.asyncio
    async def test_recommend_optimizations_success(self, performance_analytics):
        """Test successful optimization recommendations."""
        # Create performance data with poor metrics
        snapshot = MetricsSnapshot(
            aggregated_metrics={
                "avg_execution_time_seconds": 35.0,  # High execution time
                "avg_success_rate": 0.7,             # Low success rate
                "avg_throughput": 3.0                # Low throughput
            }
        )
        
        performance_data = PerformanceData(
            collection_period="24h",
            metrics_snapshots=[snapshot]
        )
        
        recommendations = await performance_analytics.recommend_optimizations(performance_data)
        
        assert len(recommendations) > 0
        
        # Check for execution time optimization
        exec_time_recs = [r for r in recommendations if r.target_metric == "execution_time_seconds"]
        assert len(exec_time_recs) > 0
        assert exec_time_recs[0].current_value == 35.0
        assert exec_time_recs[0].target_value < 35.0
        assert len(exec_time_recs[0].specific_actions) > 0
        
        # Check for success rate optimization
        success_rate_recs = [r for r in recommendations if r.target_metric == "success_rate"]
        assert len(success_rate_recs) > 0
        assert success_rate_recs[0].current_value == 0.7
        assert success_rate_recs[0].target_value > 0.7
        
        # Verify recommendations are stored
        assert len(performance_analytics.optimizations) == len(recommendations)
    
    @pytest.mark.asyncio
    async def test_recommend_optimizations_no_data(self, performance_analytics):
        """Test optimization recommendations with no data."""
        performance_data = PerformanceData(
            collection_period="24h",
            metrics_snapshots=[]
        )
        
        recommendations = await performance_analytics.recommend_optimizations(performance_data)
        assert len(recommendations) == 0
    
    @pytest.mark.asyncio
    async def test_recommend_optimizations_good_metrics(self, performance_analytics):
        """Test optimization recommendations with good metrics."""
        # Create performance data with good metrics
        snapshot = MetricsSnapshot(
            aggregated_metrics={
                "avg_execution_time_seconds": 10.0,  # Good execution time
                "avg_success_rate": 0.95,            # High success rate
                "avg_throughput": 15.0               # Good throughput
            }
        )
        
        performance_data = PerformanceData(
            collection_period="24h",
            metrics_snapshots=[snapshot]
        )
        
        recommendations = await performance_analytics.recommend_optimizations(performance_data)
        
        # Should have fewer or no recommendations for good metrics
        assert len(recommendations) == 0
    
    def test_get_real_time_metrics(self, performance_analytics):
        """Test getting real-time metrics."""
        # Add some real-time data
        now = datetime.utcnow()
        old_time = now - timedelta(hours=2)
        
        performance_analytics.real_time_metrics["execution_time"].extend([
            (old_time, 10.0),
            (now - timedelta(minutes=30), 15.0),
            (now - timedelta(minutes=10), 12.0)
        ])
        
        # Get metrics within 1 hour window
        metrics = performance_analytics.get_real_time_metrics("execution_time", window_minutes=60)
        
        # Should exclude the 2-hour old data
        assert len(metrics) == 2
        assert metrics[0][1] == 15.0
        assert metrics[1][1] == 12.0
    
    def test_get_real_time_metrics_nonexistent(self, performance_analytics):
        """Test getting real-time metrics for non-existent metric."""
        metrics = performance_analytics.get_real_time_metrics("nonexistent_metric")
        assert len(metrics) == 0
    
    def test_get_performance_summary_no_data(self, performance_analytics):
        """Test getting performance summary with no data."""
        summary = performance_analytics.get_performance_summary()
        assert summary["status"] == "no_data"
    
    def test_get_performance_summary_with_data(self, performance_analytics):
        """Test getting performance summary with data."""
        # Add some metrics
        for i in range(15):
            metrics = PerformanceMetrics(
                session_id=f"session-{i}",
                agent_configuration_id="agent-1",
                execution_time_seconds=10.0 + i,
                success_rate=0.9 - (i * 0.01),
                throughput=5.0 + i
            )
            performance_analytics.metrics_history.append(metrics)
        
        summary = performance_analytics.get_performance_summary()
        
        assert summary["status"] == "active"
        assert summary["total_metrics_collected"] == 15
        
        # Check recent performance (last 10 metrics)
        recent_perf = summary["recent_performance"]
        assert "avg_execution_time" in recent_perf
        assert "avg_success_rate" in recent_perf
        assert "avg_throughput" in recent_perf
        
        # Values should be averages of last 10 metrics (indices 5-14)
        expected_exec_time = statistics.mean([15.0 + i for i in range(10)])  # 10+5 to 10+14
        expected_success_rate = statistics.mean([0.85 - (i * 0.01) for i in range(10)])  # 0.9-0.05 to 0.9-0.14
        
        assert abs(recent_perf["avg_execution_time"] - expected_exec_time) < 0.01
        assert abs(recent_perf["avg_success_rate"] - expected_success_rate) < 0.01
        
        assert "last_updated" in summary


class TestPerformanceAnalyticsModels:
    """Test cases for performance analytics data models."""
    
    def test_metrics_collector_creation(self):
        """Test MetricsCollector model creation."""
        collector = MetricsCollector(
            name="Test Collector",
            description="A test collector",
            metric_type="performance",
            collection_interval_seconds=60,
            aggregation_method="average",
            query_pattern="execution_time.*",
            enabled=True
        )
        
        assert collector.name == "Test Collector"
        assert collector.description == "A test collector"
        assert collector.metric_type == "performance"
        assert collector.collection_interval_seconds == 60
        assert collector.aggregation_method == "average"
        assert collector.query_pattern == "execution_time.*"
        assert collector.enabled is True
    
    def test_trend_analyzer_creation(self):
        """Test TrendAnalyzer model creation."""
        analyzer = TrendAnalyzer(
            name="Test Analyzer",
            description="A test analyzer",
            metric_names=["execution_time", "success_rate"],
            analysis_window_hours=24,
            trend_detection_method="linear_regression",
            threshold_config={"significant_change": 0.1},
            alert_conditions=["success_rate < 0.8"]
        )
        
        assert analyzer.name == "Test Analyzer"
        assert analyzer.description == "A test analyzer"
        assert len(analyzer.metric_names) == 2
        assert analyzer.analysis_window_hours == 24
        assert analyzer.trend_detection_method == "linear_regression"
        assert analyzer.threshold_config["significant_change"] == 0.1
        assert len(analyzer.alert_conditions) == 1
    
    def test_optimization_recommender_creation(self):
        """Test OptimizationRecommender model creation."""
        recommender = OptimizationRecommender(
            name="Test Recommender",
            description="A test recommender",
            target_metrics=["execution_time", "throughput"],
            optimization_strategy="performance",
            recommendation_rules=["IF execution_time > 30 THEN optimize"],
            confidence_threshold=0.8
        )
        
        assert recommender.name == "Test Recommender"
        assert recommender.description == "A test recommender"
        assert len(recommender.target_metrics) == 2
        assert recommender.optimization_strategy == "performance"
        assert len(recommender.recommendation_rules) == 1
        assert recommender.confidence_threshold == 0.8
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics model creation."""
        metrics = PerformanceMetrics(
            session_id="session-1",
            agent_configuration_id="agent-1",
            execution_time_seconds=15.5,
            success_rate=0.85,
            throughput=8.0,
            error_rate=0.15,
            resource_utilization={"cpu": 45.0, "memory": 512.0},
            quality_scores={"accuracy": 0.9},
            business_metrics={"cost": 0.05},
            custom_metrics={"custom": "value"}
        )
        
        assert metrics.session_id == "session-1"
        assert metrics.agent_configuration_id == "agent-1"
        assert metrics.execution_time_seconds == 15.5
        assert metrics.success_rate == 0.85
        assert metrics.throughput == 8.0
        assert metrics.error_rate == 0.15
        assert metrics.resource_utilization["cpu"] == 45.0
        assert metrics.quality_scores["accuracy"] == 0.9
        assert metrics.business_metrics["cost"] == 0.05
        assert metrics.custom_metrics["custom"] == "value"
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_snapshot_creation(self):
        """Test MetricsSnapshot model creation."""
        metrics = [
            PerformanceMetrics(
                session_id="session-1",
                agent_configuration_id="agent-1",
                execution_time_seconds=10.0,
                success_rate=0.9,
                throughput=5.0
            )
        ]
        
        snapshot = MetricsSnapshot(
            metrics=metrics,
            aggregated_metrics={"avg_execution_time": 10.0}
        )
        
        assert len(snapshot.metrics) == 1
        assert snapshot.aggregated_metrics["avg_execution_time"] == 10.0
        assert isinstance(snapshot.timestamp, datetime)
    
    def test_trend_analysis_creation(self):
        """Test TrendAnalysis model creation."""
        analysis = TrendAnalysis(
            analyzer_name="Test Analyzer",
            metric_name="execution_time",
            analysis_period="24h",
            trend_direction="increasing",
            trend_strength=0.7,
            statistical_significance=0.85,
            key_insights=["Performance is degrading"],
            anomalies_detected=[{"timestamp": "2024-01-01", "value": 100.0}],
            predictions={"next_hour": 25.0},
            confidence_level=0.8
        )
        
        assert analysis.analyzer_name == "Test Analyzer"
        assert analysis.metric_name == "execution_time"
        assert analysis.analysis_period == "24h"
        assert analysis.trend_direction == "increasing"
        assert analysis.trend_strength == 0.7
        assert analysis.statistical_significance == 0.85
        assert len(analysis.key_insights) == 1
        assert len(analysis.anomalies_detected) == 1
        assert analysis.predictions["next_hour"] == 25.0
        assert analysis.confidence_level == 0.8
        assert isinstance(analysis.timestamp, datetime)
    
    def test_optimization_creation(self):
        """Test Optimization model creation."""
        optimization = Optimization(
            recommender_name="Test Recommender",
            target_metric="execution_time",
            current_value=30.0,
            target_value=15.0,
            improvement_percentage=50.0,
            recommendation_type="configuration",
            specific_actions=["Increase timeout", "Enable caching"],
            estimated_impact={"time_saved": 15.0},
            implementation_complexity="medium",
            confidence_score=0.8,
            priority="high"
        )
        
        assert optimization.recommender_name == "Test Recommender"
        assert optimization.target_metric == "execution_time"
        assert optimization.current_value == 30.0
        assert optimization.target_value == 15.0
        assert optimization.improvement_percentage == 50.0
        assert optimization.recommendation_type == "configuration"
        assert len(optimization.specific_actions) == 2
        assert optimization.estimated_impact["time_saved"] == 15.0
        assert optimization.implementation_complexity == "medium"
        assert optimization.confidence_score == 0.8
        assert optimization.priority == "high"
        assert isinstance(optimization.timestamp, datetime)
    
    def test_performance_data_creation(self):
        """Test PerformanceData model creation."""
        snapshot = MetricsSnapshot(
            aggregated_metrics={"avg_execution_time": 10.0}
        )
        
        analysis = TrendAnalysis(
            analyzer_name="Test",
            metric_name="test",
            analysis_period="24h",
            trend_direction="stable",
            trend_strength=0.1,
            statistical_significance=0.5,
            confidence_level=0.7
        )
        
        performance_data = PerformanceData(
            collection_period="24h",
            agent_configurations=["agent-1", "agent-2"],
            metrics_snapshots=[snapshot],
            trend_analyses=[analysis],
            benchmark_comparisons={"baseline": 10.0}
        )
        
        assert performance_data.collection_period == "24h"
        assert len(performance_data.agent_configurations) == 2
        assert len(performance_data.metrics_snapshots) == 1
        assert len(performance_data.trend_analyses) == 1
        assert performance_data.benchmark_comparisons["baseline"] == 10.0