"""Tests for InMemoryExecutor implementation."""

import asyncio
import pandas as pd
import pytest

from src.dataqa.primitives.in_memory_executor import InMemoryExecutor
from src.dataqa.exceptions import ExecutionError
from src.dataqa.models.execution import ExecutionResult


class TestInMemoryExecutor:
    """Test cases for InMemoryExecutor implementation."""
    
    @pytest.fixture
    def executor(self):
        """Create an InMemoryExecutor instance for testing."""
        return InMemoryExecutor()
    
    @pytest.fixture
    def executor_with_data(self):
        """Create an InMemoryExecutor with sample data loaded."""
        executor = InMemoryExecutor()
        
        # Load sample data
        sample_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000]
        })
        executor.load_dataframe(sample_df, 'employees')
        
        return executor
    
    def test_initialization(self, executor):
        """Test executor initialization."""
        assert executor.database_path == ":memory:"
        assert executor._connection is None
        assert isinstance(executor._python_globals, dict)
        assert 'pd' in executor._python_globals
        assert 'pandas' in executor._python_globals
    
    def test_connection_property(self, executor):
        """Test that connection property creates and returns connection."""
        # First access should create connection
        conn1 = executor.connection
        assert conn1 is not None
        
        # Second access should return same connection
        conn2 = executor.connection
        assert conn1 is conn2
    
    def test_context_manager(self):
        """Test executor as context manager."""
        with InMemoryExecutor() as executor:
            assert executor._connection is None
            # Access connection to create it
            conn = executor.connection
            assert conn is not None
        
        # Connection should be closed after context exit
        assert executor._connection is None


class TestSQLExecution:
    """Test SQL execution functionality."""
    
    @pytest.fixture
    def executor_with_data(self):
        """Create an InMemoryExecutor with sample data loaded."""
        executor = InMemoryExecutor()
        
        # Load sample data
        sample_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000]
        })
        executor.load_dataframe(sample_df, 'employees')
        
        return executor
    
    @pytest.mark.asyncio
    async def test_simple_select_query(self, executor_with_data):
        """Test executing a simple SELECT query."""
        result = await executor_with_data.execute_sql("SELECT * FROM employees LIMIT 2")
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.error is None
        assert result.output_type == "dataframe"
        assert "dataframe" in result.data
        assert len(result.data["dataframe"]) == 2
        assert result.data["columns"] == ['id', 'name', 'age', 'salary']
        assert result.data["shape"] == [2, 4]
    
    @pytest.mark.asyncio
    async def test_aggregation_query(self, executor_with_data):
        """Test executing an aggregation query."""
        result = await executor_with_data.execute_sql(
            "SELECT AVG(age) as avg_age, COUNT(*) as count FROM employees"
        )
        
        assert result.success is True
        assert len(result.data["dataframe"]) == 1
        assert "avg_age" in result.data["columns"]
        assert "count" in result.data["columns"]
        assert result.data["dataframe"][0]["count"] == 5
    
    @pytest.mark.asyncio
    async def test_parameterized_query(self, executor_with_data):
        """Test executing a parameterized query."""
        result = await executor_with_data.execute_sql(
            "SELECT * FROM employees WHERE age > ?", 
            parameters=[30]
        )
        
        assert result.success is True
        assert len(result.data["dataframe"]) == 2  # Charlie (35) and Eve (32)
    
    @pytest.mark.asyncio
    async def test_invalid_sql(self, executor_with_data):
        """Test handling of invalid SQL."""
        result = await executor_with_data.execute_sql("INVALID SQL QUERY")
        
        assert result.success is False
        assert result.error is not None
        assert "validation failed" in result.error
    
    @pytest.mark.asyncio
    async def test_dangerous_sql_blocked(self, executor_with_data):
        """Test that dangerous SQL operations are blocked."""
        dangerous_queries = [
            "DROP TABLE employees",
            "DELETE FROM employees",
            "INSERT INTO employees VALUES (6, 'Hacker', 25, 50000)",
            "UPDATE employees SET salary = 100000",
            "CREATE TABLE malicious (id INT)"
        ]
        
        for query in dangerous_queries:
            result = await executor_with_data.execute_sql(query)
            assert result.success is False
            assert "validation failed" in result.error


class TestPythonExecution:
    """Test Python code execution functionality."""
    
    @pytest.fixture
    def executor_with_data(self):
        """Create an InMemoryExecutor with sample data loaded."""
        executor = InMemoryExecutor()
        
        # Load sample data
        sample_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000]
        })
        executor.load_dataframe(sample_df, 'employees')
        
        return executor
    
    @pytest.mark.asyncio
    async def test_simple_python_execution(self, executor_with_data):
        """Test executing simple Python code."""
        code = """
result = 2 + 2
print(f"Result: {result}")
"""
        
        result = await executor_with_data.execute_python(code)
        
        assert result.success is True
        assert result.output_type == "python_execution"
        assert "stdout" in result.data
        assert "Result: 4" in result.data["stdout"]
    
    @pytest.mark.asyncio
    async def test_pandas_operations(self, executor_with_data):
        """Test Python code with pandas operations."""
        code = """
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(f"DataFrame shape: {df.shape}")
print(df.head())
"""
        
        result = await executor_with_data.execute_python(code)
        
        assert result.success is True
        assert "DataFrame shape: (3, 2)" in result.data["stdout"]
    
    @pytest.mark.asyncio
    async def test_database_access_from_python(self, executor_with_data):
        """Test accessing database from Python code."""
        code = """
df = connection.execute("SELECT COUNT(*) as count FROM employees").fetchdf()
print(f"Employee count: {df['count'].iloc[0]}")
"""
        
        result = await executor_with_data.execute_python(code)
        
        assert result.success is True
        assert "Employee count: 5" in result.data["stdout"]
    
    @pytest.mark.asyncio
    async def test_python_with_context(self, executor_with_data):
        """Test Python execution with context variables."""
        context = {"multiplier": 10}
        code = """
result = multiplier * 5
print(f"Result with context: {result}")
"""
        
        result = await executor_with_data.execute_python(code, context=context)
        
        assert result.success is True
        assert "Result with context: 50" in result.data["stdout"]
    
    @pytest.mark.asyncio
    async def test_python_syntax_error(self, executor_with_data):
        """Test handling of Python syntax errors."""
        code = "invalid python syntax !!!"
        
        result = await executor_with_data.execute_python(code)
        
        assert result.success is False
        assert "validation failed" in result.error
    
    @pytest.mark.asyncio
    async def test_dangerous_python_blocked(self, executor_with_data):
        """Test that dangerous Python operations are blocked."""
        dangerous_codes = [
            "import os",
            "from subprocess import call",
            "__import__('os')",
            "eval('print(1)')",
            "exec('print(1)')",
            "open('/etc/passwd', 'r')",
            "globals().__builtins__['eval']('1+1')"
        ]
        
        for code in dangerous_codes:
            result = await executor_with_data.execute_python(code)
            assert result.success is False
            # Different dangerous codes will fail at different stages
            assert result.error is not None


class TestSchemaOperations:
    """Test database schema operations."""
    
    @pytest.fixture
    def executor_with_data(self):
        """Create an InMemoryExecutor with sample data loaded."""
        executor = InMemoryExecutor()
        
        # Load sample data
        sample_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000]
        })
        executor.load_dataframe(sample_df, 'employees')
        
        return executor
    
    @pytest.mark.asyncio
    async def test_list_tables(self, executor_with_data):
        """Test listing all tables."""
        tables = await executor_with_data.list_tables()
        
        assert isinstance(tables, list)
        assert 'employees' in tables
    
    @pytest.mark.asyncio
    async def test_get_schema_all_tables(self, executor_with_data):
        """Test getting schema for all tables."""
        schema = await executor_with_data.get_schema()
        
        assert isinstance(schema, dict)
        assert "tables" in schema
        assert len(schema["tables"]) >= 1
        
        # Find employees table
        employees_table = next(
            (table for table in schema["tables"] if table["table_name"] == "employees"), 
            None
        )
        assert employees_table is not None
        assert "columns" in employees_table
        assert len(employees_table["columns"]) == 4
    
    @pytest.mark.asyncio
    async def test_get_schema_specific_table(self, executor_with_data):
        """Test getting schema for a specific table."""
        schema = await executor_with_data.get_schema("employees")
        
        assert schema["table_name"] == "employees"
        assert "columns" in schema
        assert len(schema["columns"]) == 4
        
        column_names = [col["column_name"] for col in schema["columns"]]
        assert "id" in column_names
        assert "name" in column_names
        assert "age" in column_names
        assert "salary" in column_names
    
    @pytest.mark.asyncio
    async def test_get_sample_data(self, executor_with_data):
        """Test getting sample data from a table."""
        result = await executor_with_data.get_sample_data("employees", limit=3)
        
        assert result.success is True
        assert len(result.data["dataframe"]) == 3
        assert result.data["columns"] == ['id', 'name', 'age', 'salary']
    
    @pytest.mark.asyncio
    async def test_get_sample_data_nonexistent_table(self, executor_with_data):
        """Test getting sample data from non-existent table."""
        result = await executor_with_data.get_sample_data("nonexistent")
        
        assert result.success is False
        assert "not found" in result.error


class TestCodeValidation:
    """Test code validation functionality."""
    
    @pytest.fixture
    def executor(self):
        """Create an InMemoryExecutor instance for testing."""
        return InMemoryExecutor()
    
    @pytest.mark.asyncio
    async def test_valid_sql_validation(self, executor):
        """Test validation of valid SQL."""
        valid_sql = "SELECT 1 as test"
        assert await executor.validate_code(valid_sql, "sql") is True
    
    @pytest.mark.asyncio
    async def test_invalid_sql_validation(self, executor):
        """Test validation of invalid SQL."""
        invalid_sql = "DROP TABLE users"
        assert await executor.validate_code(invalid_sql, "sql") is False
    
    @pytest.mark.asyncio
    async def test_valid_python_validation(self, executor):
        """Test validation of valid Python code."""
        valid_python = "x = 1 + 1\nprint(x)"
        assert await executor.validate_code(valid_python, "python") is True
    
    @pytest.mark.asyncio
    async def test_invalid_python_validation(self, executor):
        """Test validation of invalid Python code."""
        invalid_python = "import os"
        assert await executor.validate_code(invalid_python, "python") is False
    
    @pytest.mark.asyncio
    async def test_unknown_code_type(self, executor):
        """Test validation with unknown code type."""
        assert await executor.validate_code("test", "unknown") is False


class TestDataFrameLoading:
    """Test DataFrame loading functionality."""
    
    @pytest.fixture
    def executor(self):
        """Create an InMemoryExecutor instance for testing."""
        return InMemoryExecutor()
    
    def test_load_dataframe(self, executor):
        """Test loading a DataFrame into the database."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        
        executor.load_dataframe(df, 'test_table')
        
        # Verify the table was created
        result = executor.connection.execute("SELECT * FROM test_table").fetchdf()
        pd.testing.assert_frame_equal(result, df)
    
    def test_load_dataframe_error_handling(self, executor):
        """Test error handling when loading DataFrame fails."""
        # Try to load invalid data
        with pytest.raises(ExecutionError):
            executor.load_dataframe(None, 'invalid_table')


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def executor(self):
        """Create an InMemoryExecutor instance for testing."""
        return InMemoryExecutor()
    
    @pytest.mark.asyncio
    async def test_schema_error_handling(self, executor):
        """Test error handling in schema operations."""
        # This should work with empty database
        tables = await executor.list_tables()
        assert isinstance(tables, list)
        
        schema = await executor.get_schema()
        assert isinstance(schema, dict)
    
    def test_close_connection(self, executor):
        """Test closing the database connection."""
        # Create connection
        conn = executor.connection
        assert conn is not None
        
        # Close connection
        executor.close()
        assert executor._connection is None
    
    def test_close_without_connection(self, executor):
        """Test closing when no connection exists."""
        # Should not raise error
        executor.close()
        assert executor._connection is None


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple operations."""
    
    @pytest.fixture
    def executor(self):
        """Create an InMemoryExecutor instance for testing."""
        return InMemoryExecutor()
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, executor):
        """Test a complete workflow: load data, query, analyze."""
        # 1. Load data
        df = pd.DataFrame({
            'product': ['A', 'B', 'C', 'A', 'B'],
            'sales': [100, 200, 150, 120, 180],
            'region': ['North', 'South', 'North', 'South', 'North']
        })
        executor.load_dataframe(df, 'sales_data')
        
        # 2. Query data
        sql_result = await executor.execute_sql(
            "SELECT product, SUM(sales) as total_sales FROM sales_data GROUP BY product ORDER BY total_sales DESC"
        )
        assert sql_result.success is True
        assert len(sql_result.data["dataframe"]) == 3
        
        # 3. Analyze with Python
        python_code = """
df = connection.execute("SELECT * FROM sales_data").fetchdf()
avg_sales = df['sales'].mean()
print(f"Average sales: {avg_sales}")
"""
        python_result = await executor.execute_python(python_code)
        assert python_result.success is True
        assert "Average sales: 150.0" in python_result.data["stdout"]
        
        # 4. Get schema
        schema = await executor.get_schema("sales_data")
        assert schema["table_name"] == "sales_data"
        assert len(schema["columns"]) == 3