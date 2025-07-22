"""Tests for ExecutionResult data model."""

import pytest
from pydantic import ValidationError

from src.dataqa.models.execution import ExecutionResult


class TestExecutionResult:
    """Test cases for ExecutionResult model."""
    
    def test_execution_result_creation_with_required_fields(self):
        """Test creating an execution result with only required fields."""
        result = ExecutionResult(
            success=True,
            execution_time=1.5,
            code_executed="SELECT * FROM users"
        )
        
        assert result.success is True
        assert result.execution_time == 1.5
        assert result.code_executed == "SELECT * FROM users"
        assert result.data is None
        assert result.error is None
        assert result.output_type is None
    
    def test_execution_result_successful_execution(self):
        """Test creating a successful execution result with data."""
        data = {"rows": [{"id": 1, "name": "John"}], "count": 1}
        
        result = ExecutionResult(
            success=True,
            data=data,
            execution_time=0.5,
            code_executed="SELECT id, name FROM users LIMIT 1",
            output_type="dataframe"
        )
        
        assert result.success is True
        assert result.data == data
        assert result.error is None
        assert result.execution_time == 0.5
        assert result.code_executed == "SELECT id, name FROM users LIMIT 1"
        assert result.output_type == "dataframe"
    
    def test_execution_result_failed_execution(self):
        """Test creating a failed execution result with error."""
        result = ExecutionResult(
            success=False,
            error="Table 'nonexistent' doesn't exist",
            execution_time=0.1,
            code_executed="SELECT * FROM nonexistent"
        )
        
        assert result.success is False
        assert result.error == "Table 'nonexistent' doesn't exist"
        assert result.data is None
        assert result.execution_time == 0.1
        assert result.code_executed == "SELECT * FROM nonexistent"
    
    def test_execution_result_required_fields(self):
        """Test that required fields are validated."""
        # Missing success field
        with pytest.raises(ValidationError):
            ExecutionResult(
                execution_time=1.0,
                code_executed="SELECT 1"
            )
        
        # Missing execution_time field
        with pytest.raises(ValidationError):
            ExecutionResult(
                success=True,
                code_executed="SELECT 1"
            )
        
        # Missing code_executed field
        with pytest.raises(ValidationError):
            ExecutionResult(
                success=True,
                execution_time=1.0
            )
    
    def test_execution_result_data_types(self):
        """Test various data types in the data field."""
        # Dictionary data
        dict_data = {"key": "value", "number": 42}
        result = ExecutionResult(
            success=True,
            data=dict_data,
            execution_time=1.0,
            code_executed="test"
        )
        assert result.data == dict_data
        
        # List data
        list_data = [1, 2, 3, "string", {"nested": True}]
        result = ExecutionResult(
            success=True,
            data={"result": list_data},
            execution_time=1.0,
            code_executed="test"
        )
        assert result.data["result"] == list_data
    
    def test_execution_result_json_serialization(self):
        """Test that execution result can be serialized to JSON."""
        result = ExecutionResult(
            success=True,
            data={"rows": [{"id": 1}]},
            execution_time=2.5,
            code_executed="SELECT id FROM users",
            output_type="dataframe"
        )
        
        json_data = result.model_dump()
        
        assert json_data["success"] is True
        assert json_data["data"] == {"rows": [{"id": 1}]}
        assert json_data["execution_time"] == 2.5
        assert json_data["code_executed"] == "SELECT id FROM users"
        assert json_data["output_type"] == "dataframe"
        assert json_data["error"] is None
    
    def test_execution_result_from_dict(self):
        """Test creating execution result from dictionary."""
        data = {
            "success": False,
            "error": "Syntax error",
            "execution_time": 0.05,
            "code_executed": "SELCT * FROM users",  # intentional typo
            "output_type": None
        }
        
        result = ExecutionResult(**data)
        
        assert result.success is False
        assert result.error == "Syntax error"
        assert result.execution_time == 0.05
        assert result.code_executed == "SELCT * FROM users"
        assert result.output_type is None
    
    def test_execution_result_plot_data(self):
        """Test execution result with plot/visualization data."""
        plot_data = {
            "type": "matplotlib",
            "image_data": "base64_encoded_image_data",
            "format": "png"
        }
        
        result = ExecutionResult(
            success=True,
            data=plot_data,
            execution_time=3.2,
            code_executed="plt.plot([1,2,3]); plt.show()",
            output_type="plot"
        )
        
        assert result.success is True
        assert result.data == plot_data
        assert result.output_type == "plot"
    
    def test_execution_result_negative_execution_time(self):
        """Test that negative execution time is allowed (edge case)."""
        # While unusual, we don't enforce positive execution time
        # as there might be edge cases or measurement errors
        result = ExecutionResult(
            success=True,
            execution_time=-0.1,
            code_executed="test"
        )
        
        assert result.execution_time == -0.1