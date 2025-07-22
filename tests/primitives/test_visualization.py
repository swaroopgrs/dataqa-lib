"""Tests for visualization capabilities in executor primitives."""

import base64
import io
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.dataqa.primitives.in_memory_executor import InMemoryExecutor
from src.dataqa.models.execution import ExecutionResult


class TestVisualizationCapabilities:
    """Test visualization functionality in InMemoryExecutor."""
    
    @pytest.fixture
    def executor(self):
        """Create an InMemoryExecutor instance for testing."""
        return InMemoryExecutor()
    
    @pytest.fixture
    def sample_numeric_data(self):
        """Create sample numeric data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'z': np.random.randn(100)
        })
    
    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample categorical data for testing."""
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'value': np.random.randn(60),
            'count': np.random.randint(1, 100, 60)
        })
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.randn(50).cumsum(),
            'volume': np.random.randint(100, 1000, 50)
        })
    
    @pytest.mark.asyncio
    async def test_plot_capture_in_python_execution(self, executor):
        """Test that plots are captured when executing Python code."""
        code = """
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
"""
        
        result = await executor.execute_python(code)
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
        assert "image_data" in result.data["plot"]
        assert result.data["plot"]["format"] == "png"
        
        # Verify the image data is valid base64
        image_data = result.data["plot"]["image_data"]
        assert isinstance(image_data, str)
        assert len(image_data) > 0
        
        # Test that we can decode the base64 data
        try:
            decoded = base64.b64decode(image_data)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Failed to decode base64 image data: {e}")
    
    @pytest.mark.asyncio
    async def test_generate_visualization_auto_chart_type(self, executor, sample_numeric_data):
        """Test automatic chart type recommendation."""
        result = await executor.generate_visualization(sample_numeric_data, chart_type="auto")
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
        assert "image_data" in result.data["plot"]
    
    @pytest.mark.asyncio
    async def test_generate_bar_chart(self, executor, sample_categorical_data):
        """Test bar chart generation."""
        result = await executor.generate_visualization(sample_categorical_data, chart_type="bar")
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
        assert result.data["plot"]["format"] == "png"
    
    @pytest.mark.asyncio
    async def test_generate_line_chart(self, executor, sample_time_series_data):
        """Test line chart generation."""
        result = await executor.generate_visualization(sample_time_series_data, chart_type="line")
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
    
    @pytest.mark.asyncio
    async def test_generate_scatter_plot(self, executor, sample_numeric_data):
        """Test scatter plot generation."""
        result = await executor.generate_visualization(sample_numeric_data, chart_type="scatter")
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
    
    @pytest.mark.asyncio
    async def test_generate_histogram(self, executor, sample_numeric_data):
        """Test histogram generation."""
        result = await executor.generate_visualization(sample_numeric_data, chart_type="histogram")
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
    
    @pytest.mark.asyncio
    async def test_generate_box_plot(self, executor, sample_categorical_data):
        """Test box plot generation."""
        result = await executor.generate_visualization(sample_categorical_data, chart_type="box")
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
    
    @pytest.mark.asyncio
    async def test_generate_heatmap(self, executor, sample_numeric_data):
        """Test heatmap generation."""
        result = await executor.generate_visualization(sample_numeric_data, chart_type="heatmap")
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
    
    def test_chart_type_recommendation_single_numeric(self, executor):
        """Test chart type recommendation for single numeric column."""
        data = pd.DataFrame({'value': np.random.randn(100)})
        chart_type = executor._recommend_chart_type(data)
        assert chart_type == "histogram"
    
    def test_chart_type_recommendation_categorical_numeric(self, executor):
        """Test chart type recommendation for categorical + numeric data."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10,
            'value': np.random.randn(30)
        })
        chart_type = executor._recommend_chart_type(data)
        assert chart_type == "bar"
    
    def test_chart_type_recommendation_two_numeric(self, executor):
        """Test chart type recommendation for two numeric columns."""
        data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
        chart_type = executor._recommend_chart_type(data)
        assert chart_type == "scatter"
    
    def test_chart_type_recommendation_time_series(self, executor):
        """Test chart type recommendation for time series data."""
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'value': np.random.randn(50)
        })
        chart_type = executor._recommend_chart_type(data)
        assert chart_type == "line"
    
    def test_chart_type_recommendation_multiple_numeric(self, executor):
        """Test chart type recommendation for multiple numeric columns."""
        data = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100)
        })
        chart_type = executor._recommend_chart_type(data)
        assert chart_type == "heatmap"
    
    def test_chart_type_recommendation_empty_data(self, executor):
        """Test chart type recommendation for empty data."""
        data = pd.DataFrame()
        chart_type = executor._recommend_chart_type(data)
        assert chart_type == "bar"  # Default fallback
    
    @pytest.mark.asyncio
    async def test_visualization_with_custom_parameters(self, executor, sample_numeric_data):
        """Test visualization generation with custom parameters."""
        result = await executor.generate_visualization(
            sample_numeric_data, 
            chart_type="histogram",
            title="Custom Histogram",
            figsize=(12, 8),
            bins=30
        )
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
    
    @pytest.mark.asyncio
    async def test_visualization_error_handling(self, executor):
        """Test error handling in visualization generation."""
        # Test with invalid data
        invalid_data = "not a dataframe"
        result = await executor.generate_visualization(invalid_data, chart_type="bar")
        
        assert not result.success
        assert "error" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_plot_code_generation_bar_chart(self, executor, sample_categorical_data):
        """Test plot code generation for bar charts."""
        code = executor._generate_plot_code(sample_categorical_data, "bar")
        
        assert "plt.figure" in code
        assert "plot(kind='bar')" in code or "groupby" in code
        assert "plt.title" in code
        assert "plt.tight_layout" in code
    
    @pytest.mark.asyncio
    async def test_plot_code_generation_scatter_plot(self, executor, sample_numeric_data):
        """Test plot code generation for scatter plots."""
        code = executor._generate_plot_code(sample_numeric_data, "scatter")
        
        assert "plt.figure" in code
        assert "plt.scatter" in code
        assert "plt.title" in code
        assert "plt.tight_layout" in code
    
    @pytest.mark.asyncio
    async def test_plot_code_generation_heatmap(self, executor, sample_numeric_data):
        """Test plot code generation for heatmaps."""
        code = executor._generate_plot_code(sample_numeric_data, "heatmap")
        
        assert "plt.figure" in code
        assert "sns.heatmap" in code
        assert "corr()" in code
        assert "plt.title" in code
    
    @pytest.mark.asyncio
    async def test_multiple_plots_handling(self, executor):
        """Test handling of multiple plots in single execution."""
        code = """
# Create multiple subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax1.set_title('Sine')

ax2.plot(x, np.cos(x))
ax2.set_title('Cosine')
"""
        
        result = await executor.execute_python(code)
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
        assert "image_data" in result.data["plot"]
    
    @pytest.mark.asyncio
    async def test_seaborn_integration(self, executor, sample_categorical_data):
        """Test seaborn plotting integration."""
        code = """
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='category', y='value')
plt.title('Box Plot with Seaborn')
"""
        
        result = await executor.execute_python(code, context={"data": sample_categorical_data})
        
        assert result.success
        assert result.output_type == "plot"
        assert "plot" in result.data
    
    @pytest.mark.asyncio
    async def test_plot_cleanup(self, executor):
        """Test that plots are properly cleaned up between executions."""
        # First execution
        code1 = """
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('First Plot')
"""
        
        result1 = await executor.execute_python(code1)
        assert result1.success
        
        # Second execution should not interfere with first
        code2 = """
plt.figure()
plt.plot([1, 2, 3], [3, 2, 1])
plt.title('Second Plot')
"""
        
        result2 = await executor.execute_python(code2)
        assert result2.success
        
        # Both should have different plot data
        assert result1.data["plot"]["image_data"] != result2.data["plot"]["image_data"]
    
    def test_matplotlib_backend_setup(self, executor):
        """Test that matplotlib is properly configured for non-interactive use."""
        import matplotlib
        # The backend should be set to 'Agg' for non-interactive plotting
        assert matplotlib.get_backend() == 'Agg'
    
    @pytest.mark.asyncio
    async def test_visualization_with_sql_data(self, executor):
        """Test visualization generation with data from SQL queries."""
        # First, load some test data
        test_data = pd.DataFrame({
            'product': ['A', 'B', 'C', 'A', 'B', 'C'],
            'sales': [100, 150, 200, 120, 180, 220],
            'month': ['Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Feb']
        })
        
        executor.load_dataframe(test_data, 'sales_data')
        
        # Execute SQL to get data
        sql_result = await executor.execute_sql("SELECT product, AVG(sales) as avg_sales FROM sales_data GROUP BY product")
        assert sql_result.success
        
        # Convert SQL result back to DataFrame for visualization
        df_data = pd.DataFrame(sql_result.data['dataframe'])
        
        # Generate visualization
        viz_result = await executor.generate_visualization(df_data, chart_type="bar")
        
        assert viz_result.success
        assert viz_result.output_type == "plot"
        assert "plot" in viz_result.data


class TestVisualizationEdgeCases:
    """Test edge cases and error conditions for visualization."""
    
    @pytest.fixture
    def executor(self):
        """Create an InMemoryExecutor instance for testing."""
        return InMemoryExecutor()
    
    @pytest.mark.asyncio
    async def test_empty_dataframe_visualization(self, executor):
        """Test visualization with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = await executor.generate_visualization(empty_df, chart_type="bar")
        
        # Should handle gracefully and still produce a result
        assert result.success or "empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_single_row_dataframe(self, executor):
        """Test visualization with single row DataFrame."""
        single_row_df = pd.DataFrame({'x': [1], 'y': [2]})
        result = await executor.generate_visualization(single_row_df, chart_type="scatter")
        
        assert result.success
    
    @pytest.mark.asyncio
    async def test_all_nan_data(self, executor):
        """Test visualization with all NaN data."""
        nan_df = pd.DataFrame({'x': [np.nan, np.nan], 'y': [np.nan, np.nan]})
        result = await executor.generate_visualization(nan_df, chart_type="scatter")
        
        # Should handle gracefully
        assert result.success or "nan" in result.error.lower() or "empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_unsupported_chart_type(self, executor):
        """Test with unsupported chart type."""
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = await executor.generate_visualization(data, chart_type="unsupported_type")
        
        # Should fall back to default behavior
        assert result.success
    
    @pytest.mark.asyncio
    async def test_very_large_dataset_recommendation(self, executor):
        """Test chart recommendation with large dataset."""
        # Create a large dataset
        large_data = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10000,
            'value': np.random.randn(30000)
        })
        
        chart_type = executor._recommend_chart_type(large_data)
        # Should still recommend appropriate chart type
        assert chart_type in ['bar', 'box', 'histogram', 'scatter', 'line', 'heatmap']
    
    @pytest.mark.asyncio
    async def test_high_cardinality_categorical(self, executor):
        """Test with high cardinality categorical data."""
        high_card_data = pd.DataFrame({
            'category': [f'cat_{i}' for i in range(1000)],
            'value': np.random.randn(1000)
        })
        
        chart_type = executor._recommend_chart_type(high_card_data)
        # Should recommend box plot for high cardinality
        assert chart_type == 'box'