"""
DBC Resource Manager Implementation

Extends BaseResourceManager to use DBC S3 callable functions for asset retrieval.
Supports config_id and tenant_id for multi-tenant resource loading.
"""

import logging
from typing import Callable, Dict, List, Optional

import yaml
from pydantic import BaseModel

from dataqa.components.resource_manager.resource_manager import ResourceManager, ResourceConfig, ResourceManagerConfig
from dataqa.data_models.asset_models import ResourceType
from dataqa.dbc.errors import DBCCallableError, DBCClientError

logger = logging.getLogger(__name__)


class DBCResourceManager(ResourceManager):
    """
    Resource Manager implementation that uses DBC S3 callable functions.
    
    This class extends ResourceManager to work with the DBC service by using
    provided S3 callable functions instead of direct file system access.
    The only difference from ResourceManager is the data source (S3 vs local files).
    """
    
    def __init__(
        self,
        s3_callable: Callable,
        config_id: str,
        tenant_id: str,
        resources_config: Optional[List[Dict]] = None
    ):
        """
        Initialize DBC Resource Manager.
        
        Args:
            s3_callable: Callable function for S3 operations provided by DBC service
            config_id: Configuration identifier for the DBC service
            tenant_id: Tenant identifier for multi-tenant support
            resources_config: Optional list of resource configurations
        """
        self.s3_callable = s3_callable
        self.config_id = config_id
        self.tenant_id = tenant_id
        
        # Set up default resource configuration if not provided
        if resources_config is None:
            resources_config = [
                ResourceConfig(type=ResourceType.Schema, file_path=f"configs/{config_id}/schema.yml", api_url=""),
                ResourceConfig(type=ResourceType.Rule, file_path=f"configs/{config_id}/rules.yml", api_url=""),
                ResourceConfig(type=ResourceType.Example, file_path=f"configs/{config_id}/examples.yml", api_url=""),
            ]
        
        # Create config in the same format as parent class expects
        config = ResourceManagerConfig(
            source="yaml",
            resources=resources_config
        )
        
        # Initialize parent class - this will call our overridden load method
        super().__init__(config)

    def _call_s3_callable(self, operation: str, **kwargs) -> Dict:
        """
        Call the S3 callable function with error handling.
        
        Args:
            operation: The S3 operation to perform
            **kwargs: Additional arguments for the S3 callable
            
        Returns:
            Dict: Response from the S3 callable
            
        Raises:
            DBCCallableError: If the S3 callable fails
        """
        try:
            # Add config_id and tenant_id to all S3 calls
            kwargs.update({
                "config_id": self.config_id,
                "tenant_id": self.tenant_id,
                "operation": operation
            })
            
            response = self.s3_callable(**kwargs)
            
            if not response or "error" in response:
                error_msg = response.get("error", "Unknown S3 callable error") if response else "No response from S3 callable"
                raise DBCCallableError(f"S3 callable failed for operation '{operation}': {error_msg}")
                
            return response
            
        except Exception as e:
            if isinstance(e, DBCCallableError):
                raise
            raise DBCCallableError(f"S3 callable error for operation '{operation}': {str(e)}")

    def _load_yaml_from_s3(self, s3_path: str) -> Dict:
        """
        Load YAML content from S3 using the DBC S3 callable.
        
        Args:
            s3_path: S3 path to the YAML file
            
        Returns:
            Dict: Parsed YAML content
        """
        try:
            response = self._call_s3_callable("get_object", s3_path=s3_path)
            
            if "content" not in response:
                raise DBCCallableError(f"No content returned from S3 for path: {s3_path}")
                
            content = response["content"]
            
            # Parse YAML content
            if isinstance(content, str):
                return yaml.safe_load(content)
            elif isinstance(content, bytes):
                return yaml.safe_load(content.decode('utf-8'))
            else:
                return content  # Assume it's already parsed
                
        except yaml.YAMLError as e:
            raise DBCClientError(f"Failed to parse YAML from S3 path {s3_path}: {str(e)}")

    def load(self) -> Dict[str, Resource]:
        """
        Load all configured resources using DBC S3 callable.
        
        Override the parent's load method to use S3 instead of file system,
        but delegate all resource processing to the parent class.
        
        Returns:
            Dict[str, Resource]: Dictionary of loaded resources keyed by type:module_name
        """
        # Load YAML data from S3 and store in raw_data for parent class processing
        for resource_config in self.config.resources:
            try:
                logger.info(f"Loading {resource_config.type.value} from S3 path: {resource_config.file_path}")
                
                # Load YAML data from S3 (using file_path as s3_path)
                resource_data_all = self._load_yaml_from_s3(resource_config.file_path)
                
                # Store raw data in the same format as parent class expects
                if resource_config.type in [ResourceType.Schema, ResourceType.Rule, ResourceType.Example]:
                    self.raw_data[f"resource:{resource_config.type.value}:"] = resource_data_all
                else:
                    self.raw_data[f"resource:{resource_config.type.value}:"] = resource_data_all
                
                logger.info(f"Successfully loaded {resource_config.type.value} from S3")
                
            except Exception as e:
                logger.error(f"Failed to load resource {resource_config.type.value} from S3 path {resource_config.file_path}: {str(e)}")
                continue
        
        # Now call parent class load method to process the loaded data
        # But we need to temporarily override the file loading part
        original_yaml_load = yaml.safe_load
        original_open = open
        
        def mock_open(file_path, mode='r'):
            # Return the already loaded data from raw_data
            for resource_config in self.config.resources:
                if resource_config.file_path == file_path:
                    resource_key = f"resource:{resource_config.type.value}:"
                    if resource_key in self.raw_data:
                        # Create a mock file object that returns our loaded data
                        import io
                        return io.StringIO(yaml.dump(self.raw_data[resource_key]))
            raise FileNotFoundError(f"No S3 data loaded for path: {file_path}")
        
        # Temporarily replace file operations
        import builtins
        original_builtin_open = builtins.open
        builtins.open = mock_open
        
        try:
            # Call parent class load method - it will use our mocked file operations
            return super().load()
        finally:
            # Restore original file operations
            builtins.open = original_builtin_open

    # get_resource method is inherited from ResourceManager parent class
    # load_schema_embedding is not needed for initial DBC release (no vector retrieval)

    def save_dataframe_to_s3(self, dataframe_data: bytes, s3_path: str) -> str:
        """
        Save dataframe data to S3 using DBC S3 callable.
        
        Args:
            dataframe_data: Serialized dataframe data
            s3_path: S3 path where to save the dataframe
            
        Returns:
            str: S3 path of the saved dataframe
            
        Raises:
            DBCCallableError: If the S3 save operation fails
        """
        try:
            response = self._call_s3_callable(
                "put_object",
                s3_path=s3_path,
                content=dataframe_data,
                content_type="application/octet-stream"
            )
            
            if "s3_path" in response:
                return response["s3_path"]
            else:
                return s3_path  # Return the original path if not provided in response
                
        except Exception as e:
            raise DBCCallableError(f"Failed to save dataframe to S3 path {s3_path}: {str(e)}")

    def load_dataframe_from_s3(self, s3_path: str) -> bytes:
        """
        Load dataframe data from S3 using DBC S3 callable.
        
        Args:
            s3_path: S3 path to the dataframe file
            
        Returns:
            bytes: Serialized dataframe data
            
        Raises:
            DBCCallableError: If the S3 load operation fails
        """
        try:
            response = self._call_s3_callable("get_object", s3_path=s3_path)
            
            if "content" not in response:
                raise DBCCallableError(f"No content returned from S3 for dataframe path: {s3_path}")
                
            return response["content"]
            
        except Exception as e:
            raise DBCCallableError(f"Failed to load dataframe from S3 path {s3_path}: {str(e)}")