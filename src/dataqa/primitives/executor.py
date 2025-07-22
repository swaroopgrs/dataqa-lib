"""Abstract base class for executor primitive implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..exceptions import ExecutionError
from ..models.execution import ExecutionResult


class ExecutorPrimitive(ABC):
    """Abstract base class for code execution environments.
    
    This interface defines the contract for components that execute
    generated SQL and Python code in secure, controlled environments.
    Implementations might use in-memory databases, remote APIs,
    or containerized execution environments.
    """
    
    @abstractmethod
    async def execute_sql(
        self, 
        sql: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute SQL code and return results.
        
        Args:
            sql: The SQL query to execute
            parameters: Optional parameters for parameterized queries
            
        Returns:
            ExecutionResult containing the query results or error information
            
        Raises:
            ExecutionError: If execution fails due to system issues
        """
        pass
    
    @abstractmethod
    async def execute_python(
        self, 
        code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute Python code and return results.
        
        Args:
            code: The Python code to execute
            context: Optional context variables to make available to the code
            
        Returns:
            ExecutionResult containing the execution results or error information
            
        Raises:
            ExecutionError: If execution fails due to system issues
        """
        pass
    
    @abstractmethod
    async def get_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get database schema information.
        
        Args:
            table_name: Optional specific table name, if None returns all tables
            
        Returns:
            Dictionary containing schema information (tables, columns, types, etc.)
            
        Raises:
            ExecutionError: If schema retrieval fails
        """
        pass
    
    @abstractmethod
    async def list_tables(self) -> List[str]:
        """List all available tables in the database.
        
        Returns:
            List of table names
            
        Raises:
            ExecutionError: If table listing fails
        """
        pass
    
    @abstractmethod
    async def get_sample_data(
        self, 
        table_name: str, 
        limit: int = 5
    ) -> ExecutionResult:
        """Get sample data from a table.
        
        Args:
            table_name: Name of the table to sample
            limit: Maximum number of rows to return
            
        Returns:
            ExecutionResult containing sample data
            
        Raises:
            ExecutionError: If sampling fails
        """
        pass
    
    @abstractmethod
    async def validate_code(self, code: str, code_type: str) -> bool:
        """Validate code before execution for security and syntax.
        
        Args:
            code: The code to validate
            code_type: Type of code ('sql' or 'python')
            
        Returns:
            True if code is valid and safe to execute
            
        Raises:
            ExecutionError: If validation fails
        """
        pass
    
    @abstractmethod
    async def generate_visualization(
        self, 
        data: Any, 
        chart_type: str = "auto",
        **kwargs
    ) -> ExecutionResult:
        """Generate a visualization based on data characteristics.
        
        Args:
            data: Data to visualize (typically a pandas DataFrame)
            chart_type: Type of chart to generate ('auto', 'bar', 'line', 'scatter', 'histogram', 'box', 'heatmap')
            **kwargs: Additional parameters for customization
            
        Returns:
            ExecutionResult containing the generated plot
            
        Raises:
            ExecutionError: If visualization generation fails
        """
        pass