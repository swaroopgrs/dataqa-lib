import logging
import pickle
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel

from dataqa.data_models.asset_models import (
    Example,
    Resource,
    ResourceType,
    Rule,
    TableSchema,
    VectorSchema,
)
from dataqa.utils.ingestion import SchemaUtil

logger = logging.getLogger(__name__)


class ResourceConfig(BaseModel):
    type: ResourceType
    file_path: str
    api_url: str


class ResourceManagerConfig(BaseModel):
    source: Literal["yaml", "api"]
    resources: List[ResourceConfig]


class BaseResourceManager(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_resource(
        self, resource_type: ResourceType, module_name: str
    ) -> Optional[Resource]:
        pass


class ResourceManager(BaseResourceManager):
    resources: Dict[str, Resource] = {}
    config_base_model = ResourceManagerConfig

    def __init__(self, config: Union[ResourceManagerConfig, Dict]):
        """
        Initialize the resource manager.
        
        Args:
           config (Union[ResourceManagerConfig, Dict]): Either a ResourceManagerConfig object or a dictionary.
        
        Returns:
           None

        """
        self.config = config
        if isinstance(config, Dict):
            self.config = self.config_base_model(**config)
        self.raw_data = {}
        self.resources = self.load()

    def load(self):
        resources = {}
        for resource_config in self.config.resources:
            if self.config.source == "yaml":
                resource_data_all = yaml.safe_load(
                    open(resource_config.file_path, "r")
                )
                if resource_config.type == ResourceType.Schema:
                    resource_data = {
                        "type": resource_config.type,
                        "module_name": "",
                        "module_type": "",
                        "formatter": "",
                    }
                    parsed_data_list = []
                    for table_data in resource_data_all["tables"]:
                        parsed_data_list.append(TableSchema(**table_data))
                    resource_data["data"] = parsed_data_list
                    resource = Resource(**resource_data)
                    resources[
                        f"{resource_config.type.value}:{resource.module_name}"
                    ] = resource
                else:
                    for resource_data in resource_data_all["data"]:
                        resource_data["type"] = resource_config.type
                        data_list = resource_data["data"]
                        parsed_data_list = []
                        for data in data_list:
                            if resource_config.type == ResourceType.Rule:
                                parsed_data_list.append(Rule(**data))
                            elif resource_config.type == ResourceType.Example:
                                parsed_data_list.append(Example(**data))
                        resource_data["data"] = parsed_data_list
                        resource = Resource(**resource_data)
                        resources[
                            f"{resource_config.type.value}:{resource.module_name}"
                        ] = resource
                self.raw_data[
                    f"resource:{resource_config.type.value}:{resource.module_name}"
                ] = resource_data_all
        return resources

    def load_schema_embedding(self, data_file_path: str) -> None:
        """
        Load schema embedding from file.

        Args:
           data_file_path (str): Path to the file that contains the schema embedding.
        
        Returns:
           None
        
        This function loads the schema embedding from the given file path.
        If the file path is None, it loads the schema embedding from the raw data.
        The schema embedding is expected to be in either yaml or pkl format.
        After loading the schema embedding, it is converted to a list of VectorSchema objects.
        The list is then added to the data qa resources as a Resource object.
        The resource is then added to the resources dictionary.

        """
        start_time = time.time()
        print(f"Start loading schema embedding from {data_file_path}")
        if data_file_path is not None:
            if data_file_path.endswith(".yml"):
                schema_embedding = yaml.safe_load(open(data_file_path, "r"))
            elif data_file_path.endswith(".pkl"):
                schema_embedding = pickle.load(open(data_file_path, "rb"))
        else:
            schema_util = SchemaUtil()
            schema_dict = self.raw_data[f"{ResourceType.Schema.value}:"]
            schema_util.load_schema(
                schema_dict, schema_file_path=None
                )
            schema_util.parse_schema()
            schema_embedding = schema_util.parsed_schema_to_json()

        schema_embedding_list = []
        for se in schema_embedding:
            schema_embedding_list.append(VectorSchema(**se))
        schema_embedding_resource = Resource(
            data=schema_embedding_list,
            type=ResourceType.VectorSchema,
            module_name="",
            module_type="",
            formatter="",
        )
        self.resources[
            f"{schema_embedding_resource.type.value}:{schema_embedding_resource.module_name}"
        ] = schema_embedding_resource
        load_time = time.time() - start_time
        print(f"Load schema embedding: {load_time}")

    def get_resource(
        self, resource_type: ResourceType, module_name: str
    ) -> Optional[Resource]:
        """
        Retrieves a resource of the specified type and module name.

        Parameters:
           resource_type (ResourceType): The type of the resource to retrieve.
           module_name (str): The name of the module to retrieve.

        Returns:
           Optional[Resource]: The retrieved resource, or None if no resource of the specified type and module name is found.
        """
        if resource_type.value in ["schema", "vector_schema"]:
            resource_name = f"{resource_type.value}:"
        else:
            resource_name = f"{resource_type.value}:{module_name}"
        if resource_name in self.resources:
            return self.resources[resource_name]
        return None


if __name__ == "__main__":
    config = yaml.safe_load(
        open("examples/ccb_risk/config/config_ccb_risk.yml", "r")
    )
    resource_manager = ResourceManager(
        config["components"][4]["params"]["config"]
    )
    resource_manager.load_schema_embedding(
        data_file_path="examples/ccb_risk/data/schema_embedding.pkl"
    )
    resource_to_retrieve = resource_manager.get_resource(
       resource_type=ResourceType.Rule,
       module_name="code_generator",
    )
    print("Resources loaded:")
    print(resource_manager.resources.keys())