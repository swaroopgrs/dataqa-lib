"""
Advanced performance analytics for multi-agent system monitoring and optimization.
"""

import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import uuid4
from collections import defaultdict, deque

from pydantic import BaseModel, Field

from ...logging_config import get_primitive_logger
from ...exceptions import DataQAError
from ..models import ExecutionSession, ExecutionMetrics, AgentConfiguration


class MetricsCollector(BaseModel):
    """Configuration for metrics collection."""
    collector_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    metric_type: str  # "performance", "quality", "resource", "business"
    collection_interval_seconds: int = 30
    aggregation_method: str = "average"  # average, sum, max, min, percentile
    query_pattern: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrendAnalyzer(BaseModel):
    """Configuration for trend analysis."""
    analyzer_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    metric_names: List[str] = Field(default_factory=list)
    analysis_window_hours: int = 24
    trend_detection_method: str = "linear_regression"  # linear_regression, moving_average, seasonal
    threshold_config: Dict[str, float] = Field(default_factory=dict)
    alert_conditions: List[str] = Field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationRecommender(BaseModel):
    """Configuration for optimization recommendations."""
    recommender_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    target_metrics: List[str] = Field(default_factory=list)
    optimization_strategy: str = "performance"  # performance, cost, quality, balanced
    recommendation_rules: List[str] = Field(default_factory=list)
    confidence_threshold: float = 0.7
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics for agent execution."""
    metrics_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    agent_configuration_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core performance metrics
    execution_time_seconds: float
    success_rate: float
    throughput: float
    error_rate: float = 0.0
    
    # Resource utilization metrics
    resource_utilization: Dict[str, float] = Field(default_factory=dict)
    
    # Quality metrics
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Business metrics
    business_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricsSnapshot(BaseModel):
    """Snapshot of metrics at a specific point in time."""
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: List[PerformanceMetrics] = Field(default_factory=list)
    aggregated_metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrendAnalysis(BaseModel):
    """Result of trend analysis."""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    analyzer_name: str
    metric_name: str
    analysis_period: str
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    trend_strength: float  # 0.0 to 1.0
    statistical_significance: float
    key_insights: List[str] = Field(default_factory=list)
    anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list)
    predictions: Dict[str, float] = Field(default_factory=dict)
    confidence_level: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Optimization(BaseModel):
    """Optimization recommendation."""
    optimization_id: str = Field(default_factory=lambda: str(uuid4()))
    recommender_name: str
    target_metric: str
    current_value: float
    target_value: float
    improvement_percentage: float
    recommendation_type: str  # "configuration", "resource", "algorithm", "workflow"
    specific_actions: List[str] = Field(default_factory=list)
    estimated_impact: Dict[str, float] = Field(default_factory=dict)
    implementation_complexity: str = "medium"  # low, medium, high
    confidence_score: float
    priority: str = "medium"  # low, medium, high, critical
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceData(BaseModel):
    """Comprehensive performance data for analysis."""
    data_id: str = Field(default_factory=lambda: str(uuid4()))
    collection_period: str
    agent_configurations: List[str] = Field(default_factory=list)
    metrics_snapshots: List[MetricsSnapshot] = Field(default_factory=list)
    trend_analyses: List[TrendAnalysis] = Field(default_factory=list)
    benchmark_comparisons: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceAnalytics:
    """
    Advanced analytics engine for performance monitoring and optimization.
    
    Supports configurable query patterns, metrics collection, trend analysis,
    and optimization recommendations for multi-agent systems.
    """
    
    def __init__(self):
        """Initialize the performance analytics engine."""
        self.logger = get_primitive_logger("analytics", "performance")
        
        # Core data storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.snapshots_history: List[MetricsSnapshot] = []
        self.trend_analyses: List[TrendAnalysis] = []
        self.optimizations: List[Optimization] = []
        
        # Configuration
        self.metrics_collectors: List[MetricsCollector] = []
        self.trend_analyzers: List[TrendAnalyzer] = []
        self.optimization_recommenders: List[OptimizationRecommender] = []
        
        # Real-time data structures
        self.real_time_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Initialize default configurations
        self._initialize_default_configurations()
    
    def _initialize_default_configurations(self) -> None:
        """Initialize default metrics collectors, analyzers, and recommenders."""
        
        # Default performance metrics collector
        perf_collector = MetricsCollector(
            name="Core Performance Metrics",
            description="Collects basic performance metrics for all agents",
            metric_type="performance",
            collection_interval_seconds=30,
            aggregation_method="average",
            query_pattern="execution_time_seconds,success_rate,throughput,error_rate"
        )
        self.add_metrics_collector(perf_collector)
        
        # Default quality metrics collector
        quality_collector = MetricsCollector(
            name="Quality Metrics",
            description="Collects quality scores and evaluation metrics",
            metric_type="quality",
            collection_interval_seconds=60,
            aggregation_method="average",
            query_pattern="quality_scores.*"
        )
        self.add_metrics_collector(quality_collector)
        
        # Default resource utilization collector
        resource_collector = MetricsCollector(
            name="Resource Utilization",
            description="Collects resource usage metrics",
            metric_type="resource",
            collection_interval_seconds=15,
            aggregation_method="max",
            query_pattern="resource_utilization.*"
        )
        self.add_metrics_collector(resource_collector)
        
        # Default trend analyzer
        trend_analyzer = TrendAnalyzer(
            name="Performance Trend Analysis",
            description="Analyzes performance trends over time",
            metric_names=["execution_time_seconds", "success_rate", "throughput"],
            analysis_window_hours=24,
            trend_detection_method="linear_regression",
            threshold_config={
                "significant_change": 0.1,
                "volatility_threshold": 0.2
            },
            alert_conditions=["success_rate < 0.8", "execution_time_seconds > 30.0"]
        )
        self.add_trend_analyzer(trend_analyzer)
        
        # Default optimization recommender
        optimizer = OptimizationRecommender(
            name="Performance Optimizer",
            description="Provides performance optimization recommendations",
            target_metrics=["execution_time_seconds", "success_rate", "throughput"],
            optimization_strategy="balanced",
            recommendation_rules=[
                "IF execution_time_seconds > 30 THEN recommend timeout optimization",
                "IF success_rate < 0.8 THEN recommend error handling improvement",
                "IF throughput < 5 THEN recommend parallelization"
            ],
            confidence_threshold=0.7
        )
        self.add_optimization_recommender(optimizer)
    
    def add_metrics_collector(self, collector: MetricsCollector) -> None:
        """Add a metrics collector configuration.
        
        Args:
            collector: Metrics collector to add
        """
        self.metrics_collectors.append(collector)
        self.logger.info(f"Added metrics collector: {collector.name}")
    
    def add_trend_analyzer(self, analyzer: TrendAnalyzer) -> None:
        """Add a trend analyzer configuration.
        
        Args:
            analyzer: Trend analyzer to add
        """
        self.trend_analyzers.append(analyzer)
        self.logger.info(f"Added trend analyzer: {analyzer.name}")
    
    def add_optimization_recommender(self, recommender: OptimizationRecommender) -> None:
        """Add an optimization recommender configuration.
        
        Args:
            recommender: Optimization recommender to add
        """
        self.optimization_recommenders.append(recommender)
        self.logger.info(f"Added optimization recommender: {recommender.name}")
    
    async def collect_metrics(self, execution_session: ExecutionSession) -> MetricsSnapshot:
        """Collect comprehensive performance metrics from an execution session.
        
        Args:
            execution_session: Execution session to collect metrics from
            
        Returns:
            Metrics snapshot containing collected metrics
        """
        try:
            collected_metrics: List[PerformanceMetrics] = []
            
            # Extract base metrics from execution session
            base_metrics = self._extract_base_metrics(execution_session)
            collected_metrics.append(base_metrics)
            
            # Apply configured collectors
            for collector in self.metrics_collectors:
                if not collector.enabled:
                    continue
                
                try:
                    additional_metrics = await self._apply_collector(
                        execution_session, collector
                    )
                    if additional_metrics:
                        collected_metrics.extend(additional_metrics)
                except Exception as e:
                    self.logger.warning(f"Collector {collector.name} failed: {e}")
            
            # Create metrics snapshot
            snapshot = MetricsSnapshot(
                metrics=collected_metrics,
                aggregated_metrics=self._aggregate_metrics(collected_metrics)
            )
            
            # Store in history
            self.metrics_history.extend(collected_metrics)
            self.snapshots_history.append(snapshot)
            
            # Update real-time metrics
            self._update_real_time_metrics(collected_metrics)
            
            self.logger.info(f"Collected {len(collected_metrics)} metrics from session {execution_session.session_id}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            raise DataQAError(f"Metrics collection failed: {e}")
    
    def _extract_base_metrics(self, execution_session: ExecutionSession) -> PerformanceMetrics:
        """Extract base performance metrics from execution session.
        
        Args:
            execution_session: Execution session to extract from
            
        Returns:
            Base performance metrics
        """
        exec_metrics = execution_session.execution_state.execution_metrics
        
        # Calculate derived metrics
        total_steps = exec_metrics.total_steps or 1
        success_rate = (exec_metrics.completed_steps / total_steps) if total_steps > 0 else 0.0
        error_rate = (exec_metrics.failed_steps / total_steps) if total_steps > 0 else 0.0
        throughput = (exec_metrics.completed_steps / exec_metrics.total_execution_time_seconds) if exec_metrics.total_execution_time_seconds > 0 else 0.0
        
        return PerformanceMetrics(
            session_id=execution_session.session_id,
            agent_configuration_id=execution_session.workflow_id,  # Using workflow_id as proxy
            execution_time_seconds=exec_metrics.total_execution_time_seconds,
            success_rate=success_rate,
            throughput=throughput,
            error_rate=error_rate,
            resource_utilization=exec_metrics.resource_utilization.copy(),
            quality_scores=exec_metrics.quality_scores.copy()
        )
    
    async def _apply_collector(
        self,
        execution_session: ExecutionSession,
        collector: MetricsCollector
    ) -> List[PerformanceMetrics]:
        """Apply a specific metrics collector.
        
        Args:
            execution_session: Execution session
            collector: Metrics collector to apply
            
        Returns:
            List of collected metrics
        """
        # This is a simplified implementation
        # In a real system, this would apply the collector's query pattern
        # and aggregation method to extract specific metrics
        
        if collector.metric_type == "business":
            # Collect business-specific metrics
            business_metrics = {
                "user_satisfaction": 0.85,
                "cost_per_query": 0.05,
                "time_to_insight": 15.0
            }
            
            return [PerformanceMetrics(
                session_id=execution_session.session_id,
                agent_configuration_id=execution_session.workflow_id,
                execution_time_seconds=0.0,
                success_rate=1.0,
                throughput=0.0,
                business_metrics=business_metrics
            )]
        
        return []
    
    def _aggregate_metrics(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """Aggregate metrics using configured methods.
        
        Args:
            metrics: List of metrics to aggregate
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not metrics:
            return {}
        
        aggregated = {}
        
        # Aggregate core metrics
        execution_times = [m.execution_time_seconds for m in metrics if m.execution_time_seconds > 0]
        success_rates = [m.success_rate for m in metrics]
        throughputs = [m.throughput for m in metrics if m.throughput > 0]
        error_rates = [m.error_rate for m in metrics]
        
        if execution_times:
            aggregated["avg_execution_time"] = statistics.mean(execution_times)
            aggregated["max_execution_time"] = max(execution_times)
            aggregated["min_execution_time"] = min(execution_times)
        
        if success_rates:
            aggregated["avg_success_rate"] = statistics.mean(success_rates)
        
        if throughputs:
            aggregated["avg_throughput"] = statistics.mean(throughputs)
        
        if error_rates:
            aggregated["avg_error_rate"] = statistics.mean(error_rates)
        
        return aggregated
    
    def _update_real_time_metrics(self, metrics: List[PerformanceMetrics]) -> None:
        """Update real-time metrics storage.
        
        Args:
            metrics: Metrics to add to real-time storage
        """
        for metric in metrics:
            # Store key metrics in real-time deques
            self.real_time_metrics["execution_time"].append(
                (datetime.utcnow(), metric.execution_time_seconds)
            )
            self.real_time_metrics["success_rate"].append(
                (datetime.utcnow(), metric.success_rate)
            )
            self.real_time_metrics["throughput"].append(
                (datetime.utcnow(), metric.throughput)
            )
    
    async def analyze_trends(self, historical_data: List[MetricsSnapshot]) -> TrendAnalysis:
        """Analyze performance trends using configured analyzers.
        
        Args:
            historical_data: Historical metrics snapshots
            
        Returns:
            Comprehensive trend analysis
        """
        try:
            # Use the first enabled trend analyzer
            analyzer = next((a for a in self.trend_analyzers if a.enabled), None)
            if not analyzer:
                self.logger.warning("No enabled trend analyzers found")
                return self._create_default_trend_analysis()
            
            # Prepare data for analysis
            trend_data = self._prepare_trend_data(historical_data, analyzer)
            
            # Perform trend analysis
            analysis = await self._perform_trend_analysis(trend_data, analyzer)
            
            # Store analysis
            self.trend_analyses.append(analysis)
            
            self.logger.info(f"Completed trend analysis: {analysis.trend_direction}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return self._create_default_trend_analysis()
    
    def _prepare_trend_data(
        self,
        historical_data: List[MetricsSnapshot],
        analyzer: TrendAnalyzer
    ) -> Dict[str, List[float]]:
        """Prepare data for trend analysis.
        
        Args:
            historical_data: Historical metrics data
            analyzer: Trend analyzer configuration
            
        Returns:
            Dictionary of metric time series data
        """
        trend_data = defaultdict(list)
        
        # Extract time series for each metric
        for snapshot in historical_data:
            for metric_name in analyzer.metric_names:
                value = snapshot.aggregated_metrics.get(f"avg_{metric_name}")
                if value is not None:
                    trend_data[metric_name].append(value)
        
        return dict(trend_data)
    
    async def _perform_trend_analysis(
        self,
        trend_data: Dict[str, List[float]],
        analyzer: TrendAnalyzer
    ) -> TrendAnalysis:
        """Perform actual trend analysis.
        
        Args:
            trend_data: Time series data
            analyzer: Analyzer configuration
            
        Returns:
            Trend analysis result
        """
        # Simple trend analysis implementation
        # In a real system, this would use more sophisticated statistical methods
        
        primary_metric = analyzer.metric_names[0] if analyzer.metric_names else "execution_time_seconds"
        data_points = trend_data.get(primary_metric, [])
        
        if len(data_points) < 2:
            return TrendAnalysis(
                analyzer_name=analyzer.name,
                metric_name=primary_metric,
                analysis_period=f"{analyzer.analysis_window_hours}h",
                trend_direction="insufficient_data",
                trend_strength=0.0,
                statistical_significance=0.0,
                confidence_level=0.0
            )
        
        # Calculate simple trend
        first_half = data_points[:len(data_points)//2]
        second_half = data_points[len(data_points)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change_ratio = (second_avg - first_avg) / first_avg if first_avg != 0 else 0
        
        if abs(change_ratio) < 0.05:
            trend_direction = "stable"
        elif change_ratio > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        trend_strength = min(abs(change_ratio), 1.0)
        
        # Generate insights
        insights = []
        if trend_direction == "increasing" and primary_metric == "execution_time_seconds":
            insights.append("Execution times are increasing - consider performance optimization")
        elif trend_direction == "decreasing" and primary_metric == "success_rate":
            insights.append("Success rates are declining - review error handling and agent logic")
        elif trend_direction == "stable":
            insights.append("Performance metrics are stable")
        
        return TrendAnalysis(
            analyzer_name=analyzer.name,
            metric_name=primary_metric,
            analysis_period=f"{analyzer.analysis_window_hours}h",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            statistical_significance=0.8,  # Simplified
            key_insights=insights,
            confidence_level=0.7
        )
    
    def _create_default_trend_analysis(self) -> TrendAnalysis:
        """Create a default trend analysis when analysis fails.
        
        Returns:
            Default trend analysis
        """
        return TrendAnalysis(
            analyzer_name="default",
            metric_name="unknown",
            analysis_period="24h",
            trend_direction="unknown",
            trend_strength=0.0,
            statistical_significance=0.0,
            key_insights=["Trend analysis unavailable"],
            confidence_level=0.0
        )
    
    async def recommend_optimizations(self, performance_data: PerformanceData) -> List[Optimization]:
        """Generate optimization recommendations based on performance data.
        
        Args:
            performance_data: Performance data to analyze
            
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations: List[Optimization] = []
            
            # Apply each enabled recommender
            for recommender in self.optimization_recommenders:
                if not recommender.enabled:
                    continue
                
                try:
                    recs = await self._apply_recommender(performance_data, recommender)
                    recommendations.extend(recs)
                except Exception as e:
                    self.logger.warning(f"Recommender {recommender.name} failed: {e}")
            
            # Sort by priority and confidence
            recommendations.sort(
                key=lambda x: (
                    {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority],
                    x.confidence_score
                ),
                reverse=True
            )
            
            # Store recommendations
            self.optimizations.extend(recommendations)
            
            self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Optimization recommendation failed: {e}")
            return []
    
    async def _apply_recommender(
        self,
        performance_data: PerformanceData,
        recommender: OptimizationRecommender
    ) -> List[Optimization]:
        """Apply a specific optimization recommender.
        
        Args:
            performance_data: Performance data
            recommender: Recommender configuration
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Get latest metrics
        if not performance_data.metrics_snapshots:
            return recommendations
        
        latest_snapshot = performance_data.metrics_snapshots[-1]
        
        # Apply recommendation rules
        for target_metric in recommender.target_metrics:
            current_value = latest_snapshot.aggregated_metrics.get(f"avg_{target_metric}")
            if current_value is None:
                continue
            
            # Generate recommendations based on simple rules
            if target_metric == "execution_time_seconds" and current_value > 30.0:
                recommendations.append(Optimization(
                    recommender_name=recommender.name,
                    target_metric=target_metric,
                    current_value=current_value,
                    target_value=15.0,
                    improvement_percentage=50.0,
                    recommendation_type="configuration",
                    specific_actions=[
                        "Increase timeout settings",
                        "Enable parallel processing",
                        "Optimize query complexity"
                    ],
                    estimated_impact={"execution_time_reduction": 0.5},
                    confidence_score=0.8,
                    priority="high"
                ))
            
            elif target_metric == "success_rate" and current_value < 0.8:
                recommendations.append(Optimization(
                    recommender_name=recommender.name,
                    target_metric=target_metric,
                    current_value=current_value,
                    target_value=0.95,
                    improvement_percentage=18.75,
                    recommendation_type="algorithm",
                    specific_actions=[
                        "Improve error handling",
                        "Add input validation",
                        "Implement retry mechanisms"
                    ],
                    estimated_impact={"success_rate_improvement": 0.15},
                    confidence_score=0.9,
                    priority="critical"
                ))
        
        return recommendations
    
    def get_real_time_metrics(self, metric_name: str, window_minutes: int = 60) -> List[tuple]:
        """Get real-time metrics for a specific metric.
        
        Args:
            metric_name: Name of metric to retrieve
            window_minutes: Time window in minutes
            
        Returns:
            List of (timestamp, value) tuples
        """
        if metric_name not in self.real_time_metrics:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        return [
            (timestamp, value) 
            for timestamp, value in self.real_time_metrics[metric_name]
            if timestamp >= cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of overall performance.
        
        Returns:
            Dictionary containing performance summary
        """
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        avg_execution_time = statistics.mean([m.execution_time_seconds for m in recent_metrics])
        avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
        avg_throughput = statistics.mean([m.throughput for m in recent_metrics if m.throughput > 0])
        
        return {
            "status": "active",
            "total_metrics_collected": len(self.metrics_history),
            "recent_performance": {
                "avg_execution_time": avg_execution_time,
                "avg_success_rate": avg_success_rate,
                "avg_throughput": avg_throughput or 0.0
            },
            "trend_analyses_count": len(self.trend_analyses),
            "optimizations_generated": len(self.optimizations),
            "last_updated": datetime.utcnow().isoformat()
        }