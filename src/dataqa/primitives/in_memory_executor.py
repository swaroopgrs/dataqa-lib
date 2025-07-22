"""In-memory executor implementation using DuckDB."""

import ast
import asyncio
import base64
import io
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Union
import warnings

import duckdb
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..exceptions import ExecutionError
from ..models.execution import ExecutionResult
from .executor import ExecutorPrimitive


class InMemoryExecutor(ExecutorPrimitive):
    """In-memory executor using DuckDB for SQL and controlled Python execution.
    
    This implementation provides a secure, sandboxed environment for executing
    SQL queries and Python code. It uses DuckDB as the SQL engine with pandas
    integration and implements controlled Python execution with restricted
    built-ins and imports.
    """
    
    def __init__(self, config: Union[str, Dict[str, Any]] = ":memory:"):
        """Initialize the in-memory executor.
        
        Args:
            config: Either a database path string or configuration dictionary
        """
        if isinstance(config, str):
            # Legacy string path
            self.database_path = config
            self.max_execution_time = 30.0
            self.max_memory_mb = 512
        else:
            # Configuration dictionary
            self.database_path = config.get("database_path", ":memory:")
            self.max_execution_time = config.get("max_execution_time", 30.0)
            self.max_memory_mb = config.get("max_memory_mb", 512)
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._python_globals: Dict[str, Any] = {}
        self._setup_matplotlib()
        self._setup_python_environment()
    
    def _setup_matplotlib(self) -> None:
        """Set up matplotlib for non-interactive plotting."""
        # Use Agg backend for non-interactive plotting
        matplotlib.use('Agg')
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.style.use('default')
    
    def _setup_python_environment(self) -> None:
        """Set up the controlled Python execution environment."""
        # Safe built-ins for Python execution
        safe_builtins = {
            'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
            'float', 'int', 'len', 'list', 'map', 'max', 'min', 'range',
            'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip',
            'print', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr'
        }
        
        # Get built-ins properly (handle both dict and module cases)
        import builtins
        safe_builtins_dict = {}
        for name in safe_builtins:
            if hasattr(builtins, name):
                safe_builtins_dict[name] = getattr(builtins, name)
        
        # Add a restricted __import__ that only allows safe modules
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Allow only specific safe modules that pandas might need
            allowed_modules = {
                'numpy', 'pandas', 'datetime', 'decimal', 'fractions',
                'collections', 'itertools', 'functools', 'operator',
                'math', 'statistics', 'json', 're'
            }
            
            # Block dangerous imports
            if name in ['os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 'http']:
                raise ImportError(f"Import of '{name}' is not allowed")
            
            # For pandas internal imports, we'll be more permissive but still block dangerous ones
            if name.startswith('pandas') or name.startswith('numpy') or name in allowed_modules:
                return builtins.__import__(name, globals, locals, fromlist, level)
            
            raise ImportError(f"Import of '{name}' is not allowed")
        
        safe_builtins_dict['__import__'] = restricted_import
        
        # Create restricted globals for Python execution
        self._python_globals = {
            '__builtins__': safe_builtins_dict,
            'pd': pd,
            'pandas': pd,
            'plt': plt,
            'matplotlib': matplotlib,
            'sns': sns,
            'seaborn': sns,
            'np': np,
            'numpy': np,
        }
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if self._connection is None:
            self._connection = duckdb.connect(self.database_path)
            # DuckDB has built-in pandas support, no need to install extension
        return self._connection
    
    async def execute_sql(
        self, 
        sql: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute SQL code using DuckDB.
        
        Args:
            sql: The SQL query to execute
            parameters: Optional parameters for parameterized queries
            
        Returns:
            ExecutionResult containing the query results or error information
        """
        start_time = time.time()
        
        try:
            # Validate SQL before execution
            if not await self.validate_code(sql, "sql"):
                return ExecutionResult(
                    success=False,
                    error="SQL validation failed: potentially unsafe query",
                    execution_time=time.time() - start_time,
                    code_executed=sql
                )
            
            # Execute the SQL query
            if parameters:
                result = self.connection.execute(sql, parameters).fetchdf()
            else:
                result = self.connection.execute(sql).fetchdf()
            
            # Convert DataFrame to dictionary for serialization
            data = {
                "dataframe": result.to_dict(orient="records"),
                "columns": result.columns.tolist(),
                "shape": list(result.shape),  # Convert tuple to list for consistency
                "dtypes": result.dtypes.astype(str).to_dict()
            }
            
            return ExecutionResult(
                success=True,
                data=data,
                execution_time=time.time() - start_time,
                code_executed=sql,
                output_type="dataframe"
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"SQL execution error: {str(e)}",
                execution_time=time.time() - start_time,
                code_executed=sql
            )
    
    async def execute_python(
        self, 
        code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute Python code in a controlled environment.
        
        Args:
            code: The Python code to execute
            context: Optional context variables to make available to the code
            
        Returns:
            ExecutionResult containing the execution results or error information
        """
        start_time = time.time()
        
        try:
            # Validate Python code before execution
            if not await self.validate_code(code, "python"):
                return ExecutionResult(
                    success=False,
                    error="Python validation failed: potentially unsafe code",
                    execution_time=time.time() - start_time,
                    code_executed=code
                )
            
            # Set up execution environment
            execution_globals = self._python_globals.copy()
            if context:
                execution_globals.update(context)
            
            # Add connection to globals for database access
            execution_globals['connection'] = self.connection
            execution_globals['conn'] = self.connection  # Common alias
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Execute the code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile and execute the code
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, execution_globals)
            
            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Check if any plots were created
            plot_data = None
            output_type = "python_execution"
            
            if plt.get_fignums():  # Check if there are any figures
                plot_data = self._capture_plot()
                output_type = "plot"
            
            # Prepare result data
            data = {
                "stdout": stdout_output,
                "stderr": stderr_output,
                "globals": {k: str(v) for k, v in execution_globals.items() 
                           if not k.startswith('__') and k not in ['pd', 'pandas', 'connection', 'conn', 'plt', 'matplotlib', 'sns', 'seaborn', 'np', 'numpy']}
            }
            
            # Add plot data if available
            if plot_data:
                data["plot"] = plot_data
            
            return ExecutionResult(
                success=True,
                data=data,
                execution_time=time.time() - start_time,
                code_executed=code,
                output_type=output_type
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Python execution error: {str(e)}",
                execution_time=time.time() - start_time,
                code_executed=code
            )
    
    async def get_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get database schema information.
        
        Args:
            table_name: Optional specific table name, if None returns all tables
            
        Returns:
            Dictionary containing schema information
        """
        try:
            if table_name:
                # Get schema for specific table
                schema_query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = ?
                ORDER BY ordinal_position
                """
                columns_df = self.connection.execute(schema_query, [table_name]).fetchdf()
                
                return {
                    "table_name": table_name,
                    "columns": columns_df.to_dict(orient="records")
                }
            else:
                # Get schema for all tables
                tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                tables_df = self.connection.execute(tables_query).fetchdf()
                
                schema_info = {"tables": []}
                
                for table in tables_df['table_name']:
                    columns_query = """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = ?
                    ORDER BY ordinal_position
                    """
                    columns_df = self.connection.execute(columns_query, [table]).fetchdf()
                    
                    schema_info["tables"].append({
                        "table_name": table,
                        "columns": columns_df.to_dict(orient="records")
                    })
                
                return schema_info
                
        except Exception as e:
            raise ExecutionError(f"Failed to get schema: {str(e)}")
    
    async def list_tables(self) -> List[str]:
        """List all available tables in the database.
        
        Returns:
            List of table names
        """
        try:
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            result = self.connection.execute(query).fetchdf()
            return result['table_name'].tolist()
        except Exception as e:
            raise ExecutionError(f"Failed to list tables: {str(e)}")
    
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
        """
        start_time = time.time()
        
        try:
            # Validate table name to prevent SQL injection
            tables = await self.list_tables()
            if table_name not in tables:
                return ExecutionResult(
                    success=False,
                    error=f"Table '{table_name}' not found",
                    execution_time=time.time() - start_time,
                    code_executed=f"SELECT * FROM {table_name} LIMIT {limit}"
                )
            
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return await self.execute_sql(query)
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Failed to get sample data: {str(e)}",
                execution_time=time.time() - start_time,
                code_executed=f"SELECT * FROM {table_name} LIMIT {limit}"
            )
    
    async def validate_code(self, code: str, code_type: str) -> bool:
        """Validate code before execution for security and syntax.
        
        Args:
            code: The code to validate
            code_type: Type of code ('sql' or 'python')
            
        Returns:
            True if code is valid and safe to execute
        """
        try:
            if code_type.lower() == "sql":
                return self._validate_sql(code)
            elif code_type.lower() == "python":
                return self._validate_python(code)
            else:
                return False
        except Exception:
            return False
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL code for safety and syntax.
        
        Args:
            sql: SQL code to validate
            
        Returns:
            True if SQL is safe to execute
        """
        # Convert to lowercase for checking
        sql_lower = sql.lower().strip()
        
        # Block potentially dangerous SQL operations
        dangerous_keywords = [
            'drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update',
            'grant', 'revoke', 'exec', 'execute', 'sp_', 'xp_'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
        
        # Basic syntax validation by attempting to prepare the statement
        try:
            # Use DuckDB's prepare to validate syntax without execution
            # Create a temporary connection for validation to avoid side effects
            temp_conn = duckdb.connect(":memory:")
            temp_conn.execute("PREPARE stmt AS " + sql)
            temp_conn.execute("DEALLOCATE stmt")
            temp_conn.close()
            return True
        except Exception:
            # If prepare fails, try a simple syntax check by parsing
            # Allow the query if it starts with SELECT (read-only operations)
            if sql_lower.strip().startswith('select'):
                return True
            return False
    
    def _validate_python(self, code: str) -> bool:
        """Validate Python code for safety and syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            True if Python code is safe to execute
        """
        try:
            # Parse the code to check syntax
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                # Allow imports of safe modules that are already in globals
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in ['matplotlib', 'matplotlib.pyplot', 'numpy', 'pandas', 'seaborn']:
                            return False
                
                if isinstance(node, ast.ImportFrom):
                    if node.module not in ['matplotlib', 'matplotlib.pyplot', 'numpy', 'pandas', 'seaborn']:
                        return False
                
                # Block access to dangerous attributes
                if isinstance(node, ast.Attribute):
                    if node.attr in ['__import__', '__builtins__', '__globals__', '__locals__']:
                        return False
                
                # Block function calls to dangerous functions
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', 'open']:
                            return False
            
            return True
            
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def _capture_plot(self) -> Dict[str, Any]:
        """Capture the current matplotlib plot as base64 encoded image data.
        
        Returns:
            Dictionary containing plot data and metadata
        """
        try:
            # Get the current figure
            fig = plt.gcf()
            
            # Save plot to bytes buffer
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Encode as base64
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Get plot metadata
            plot_data = {
                "image_data": img_base64,
                "format": "png",
                "width": fig.get_figwidth(),
                "height": fig.get_figheight(),
                "dpi": fig.dpi
            }
            
            # Clear the figure to free memory
            plt.close(fig)
            
            return plot_data
            
        except Exception as e:
            # If plot capture fails, return error info
            return {
                "error": f"Failed to capture plot: {str(e)}",
                "image_data": None
            }
    
    async def generate_visualization(
        self, 
        data: pd.DataFrame, 
        chart_type: str = "auto",
        **kwargs
    ) -> ExecutionResult:
        """Generate a visualization based on data characteristics.
        
        Args:
            data: DataFrame to visualize
            chart_type: Type of chart to generate ('auto', 'bar', 'line', 'scatter', 'histogram', 'box', 'heatmap')
            **kwargs: Additional parameters for customization
            
        Returns:
            ExecutionResult containing the generated plot
        """
        start_time = time.time()
        
        try:
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Determine chart type if auto
            if chart_type == "auto":
                chart_type = self._recommend_chart_type(data)
            
            # Generate the appropriate visualization
            code = self._generate_plot_code(data, chart_type, **kwargs)
            
            # Execute the plotting code
            return await self.execute_python(code, context={"data": data})
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Visualization generation error: {str(e)}",
                execution_time=time.time() - start_time,
                code_executed=f"generate_visualization(chart_type='{chart_type}')"
            )
    
    def _recommend_chart_type(self, data: pd.DataFrame) -> str:
        """Recommend the best chart type based on data characteristics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Recommended chart type
        """
        if data.empty:
            return "bar"
        
        num_cols = len(data.select_dtypes(include=[np.number]).columns)
        cat_cols = len(data.select_dtypes(include=['object', 'category']).columns)
        
        # Time series detection - check for datetime columns first
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and num_cols >= 1:
            return "line"
        
        # Single numeric column
        if num_cols == 1 and cat_cols == 0:
            return "histogram"
        
        # One categorical, one numeric
        if num_cols == 1 and cat_cols == 1:
            unique_cats = data.select_dtypes(include=['object', 'category']).nunique().iloc[0]
            if unique_cats <= 10:
                return "bar"
            else:
                return "box"
        
        # Two numeric columns
        if num_cols == 2 and cat_cols == 0:
            return "scatter"
        
        # Multiple numeric columns - correlation heatmap
        if num_cols > 2:
            return "heatmap"
        
        # Default fallback
        return "bar"
    
    def _generate_plot_code(
        self, 
        data: pd.DataFrame, 
        chart_type: str, 
        **kwargs
    ) -> str:
        """Generate Python code for creating the specified chart type.
        
        Args:
            data: DataFrame to plot
            chart_type: Type of chart to generate
            **kwargs: Additional parameters
            
        Returns:
            Python code string for generating the plot
        """
        # Handle empty DataFrame
        if data.empty:
            figsize = kwargs.get('figsize', (10, 6))
            title = kwargs.get('title', 'Empty Data')
            return f"""
plt.figure(figsize={figsize})
plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
plt.title('{title}')
plt.axis('off')
plt.tight_layout()
"""
        
        # Get column information
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Extract common parameters
        title = kwargs.get('title', f'{chart_type.title()} Chart')
        figsize = kwargs.get('figsize', (10, 6))
        
        if chart_type == "bar":
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                return f"""
plt.figure(figsize={figsize})
data.groupby('{x_col}')['{y_col}'].mean().plot(kind='bar')
plt.title('{title}')
plt.xlabel('{x_col}')
plt.ylabel('{y_col}')
plt.xticks(rotation=45)
plt.tight_layout()
"""
            elif len(data.columns) > 0:
                # Fallback to value counts of first column
                col = data.columns[0]
                return f"""
plt.figure(figsize={figsize})
data['{col}'].value_counts().head(10).plot(kind='bar')
plt.title('{title}')
plt.xlabel('{col}')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
"""
            else:
                # No columns available
                return f"""
plt.figure(figsize={figsize})
plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
plt.title('{title}')
plt.axis('off')
plt.tight_layout()
"""
        
        elif chart_type == "line":
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                x_col = date_cols[0]
                y_col = numeric_cols[0]
                return f"""
plt.figure(figsize={figsize})
plt.plot(data['{x_col}'], data['{y_col}'])
plt.title('{title}')
plt.xlabel('{x_col}')
plt.ylabel('{y_col}')
plt.xticks(rotation=45)
plt.tight_layout()
"""
            elif len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                return f"""
plt.figure(figsize={figsize})
plt.plot(data['{x_col}'], data['{y_col}'])
plt.title('{title}')
plt.xlabel('{x_col}')
plt.ylabel('{y_col}')
plt.tight_layout()
"""
            else:
                # Line plot of single numeric column
                col = numeric_cols[0] if numeric_cols else data.columns[0]
                return f"""
plt.figure(figsize={figsize})
data['{col}'].plot(kind='line')
plt.title('{title}')
plt.ylabel('{col}')
plt.tight_layout()
"""
        
        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                return f"""
plt.figure(figsize={figsize})
plt.scatter(data['{x_col}'], data['{y_col}'], alpha=0.6)
plt.title('{title}')
plt.xlabel('{x_col}')
plt.ylabel('{y_col}')
plt.tight_layout()
"""
            else:
                # Fallback to histogram
                col = numeric_cols[0] if numeric_cols else data.columns[0]
                return f"""
plt.figure(figsize={figsize})
plt.hist(data['{col}'], bins=20, alpha=0.7)
plt.title('{title}')
plt.xlabel('{col}')
plt.ylabel('Frequency')
plt.tight_layout()
"""
        
        elif chart_type == "histogram":
            col = numeric_cols[0] if numeric_cols else data.columns[0]
            bins = kwargs.get('bins', 20)
            return f"""
plt.figure(figsize={figsize})
plt.hist(data['{col}'], bins={bins}, alpha=0.7, edgecolor='black')
plt.title('{title}')
plt.xlabel('{col}')
plt.ylabel('Frequency')
plt.tight_layout()
"""
        
        elif chart_type == "box":
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                return f"""
plt.figure(figsize={figsize})
sns.boxplot(data=data, x='{categorical_cols[0]}', y='{numeric_cols[0]}')
plt.title('{title}')
plt.xticks(rotation=45)
plt.tight_layout()
"""
            else:
                col = numeric_cols[0] if numeric_cols else data.columns[0]
                return f"""
plt.figure(figsize={figsize})
plt.boxplot(data['{col}'].dropna())
plt.title('{title}')
plt.ylabel('{col}')
plt.tight_layout()
"""
        
        elif chart_type == "heatmap":
            if len(numeric_cols) > 1:
                return f"""
plt.figure(figsize={figsize})
correlation_matrix = data[{numeric_cols}].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('{title}')
plt.tight_layout()
"""
            else:
                # Fallback to bar chart
                col = data.columns[0]
                return f"""
plt.figure(figsize={figsize})
data['{col}'].value_counts().head(10).plot(kind='bar')
plt.title('{title}')
plt.xlabel('{col}')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
"""
        
        else:
            # Default fallback
            col = data.columns[0]
            return f"""
plt.figure(figsize={figsize})
data['{col}'].value_counts().head(10).plot(kind='bar')
plt.title('{title}')
plt.xlabel('{col}')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
"""

    async def load_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Load a pandas DataFrame into the database as a table.
        
        Args:
            table_name: Name for the table
            df: DataFrame to load
        """
        try:
            self.connection.register(table_name, df)
        except Exception as e:
            raise ExecutionError(f"Failed to load DataFrame: {str(e)}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()