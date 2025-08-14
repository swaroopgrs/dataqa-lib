import itertools
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from dataqa.components.base_component import (
    Component,
    ComponentConfig,
)
from dataqa.components.resource_manager.resource_manager import (
    Resource,
    ResourceManager,
    ResourceType,
)
from dataqa.data_models.asset_models import RetrievedAsset
from dataqa.utils.data_model_util import create_base_model
from dataqa.utils.schema_util import (
    convert_table_schema_to_sql_str,
    reconstruct_table_schema,
)

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    TAG = "tag"
    VECTOR = "vector"
    HYBRID = "hybrid"
    ALL = "all"


class RetrieverInput(BaseModel):
    query: Any = Field(description="Query for retrieving the asset")


class RetrieverConfig(ComponentConfig):
    resource_type: List[ResourceType] = Field(
        description="Resource type. Values: rule, schema, example"
    )
    module_name: List[str] = Field(
        description="retrieve resource for this module name"
    )
    retrieval_method: RetrievalMethod = Field(
        description="Retrieval algorithm or method"
    )
    parameters: Dict[str, Any] = Field(
        description="parameters of retriever component"
    )
    # top_k: int = Field(default=5, description="Default number of top assets to retrieve")


class RetrieverOutput(BaseModel):
    output_data: List[RetrievedAsset] = Field(
        description="list of retrieved assets"
    )
    # output_text: str = Field(description="Text string to be inserted to the prompt")
    # retrieval_details: Dict[str, Any] = Field(default_factory=dict, description="Could contain retrieval details")
    # component_type: str = Field(description="Retriever Component Type")


class Retriever(Component, ABC):
    config: RetrieverConfig
    retrieval_method: RetrievalMethod
    parameters: Dict[str, Any]

    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self.retrieval_method = self.config.retrieval_method
        self.parameters = self.config.parameters

    @abstractmethod
    def retrieve_assets(
        self, query: Any, resource: Resource
    ) -> List[RetrievedAsset]:
        pass

    async def run(
        self, input_data: RetrieverInput, config: Dict = {}
    ) -> RetrieverOutput:
        pass

    @staticmethod
    def prepare_output_string(
        retrieved_asset: List[RetrievedAsset], resource: Resource
    ) -> str:
        output_str_list = []
        if resource.type.value == "vector_schema":
            reconstructed_table = reconstruct_table_schema(
                retrieved_asset, resource
            )
            schema_str = "\n".join(
                [
                    convert_table_schema_to_sql_str(t.dict())
                    for t in reconstructed_table.data
                ]
            )
            output_str_list.append(schema_str)
        else:
            for r in retrieved_asset:
                if resource.type.value == "rule":
                    output_str_list.append(
                        resource.formatter.format(**r.content.dict())
                    )
                elif resource.type.value == "example":
                    if isinstance(r.content.example, dict):
                        output_str_list.append(
                            resource.formatter.format(**r.content.example)
                        )
                    elif isinstance(r.content.example, str):
                        output_str_list.append(r.content.example)
                elif resource.type.value == "schema":
                    schema_str = convert_table_schema_to_sql_str(
                        r.content.dict()
                    )
                    output_str_list.append(schema_str)
        output_str = "\n".join(output_str_list)
        return output_str

    @staticmethod
    def create_output_config(
        resource_type_list: List[ResourceType],
        module_name_list: List[str],
        resource_manager: ResourceManager,
    ) -> List[Dict]:
        output_config = []
        for resource_type, module_name in list(
            itertools.product(resource_type_list, module_name_list)
        ):
            resource = resource_manager.get_resource(resource_type, module_name)
            if resource is None:
                continue
            output_config.append(
                {
                    "name": Retriever.get_state_name(
                        resource_type, module_name
                    ),
                    "type": "str",
                    "description": f"Retrieved {resource_type.value} section in prompt for {module_name} module.",
                }
            )
        return output_config

    @staticmethod
    def get_state_name(resource_type: ResourceType, module_name: str) -> str:
        return f"{module_name}_{resource_type.value}"


class AllRetriever(Retriever):
    component_type = "AllRetriever"
    config_base_model = RetrieverConfig
    input_base_model = RetrieverInput
    output_base_model = "dynamically built"

    def __init__(
        self,
        config: Union[Dict, RetrieverConfig],
        resource_manager: ResourceManager,
    ):
        retriever_config = config
        if isinstance(retriever_config, Dict):
            retriever_config = RetrieverConfig.model_validate(config)
        super().__init__(retriever_config)
        self.resource_manager = resource_manager
        
        output_base_model_name = f"{self.config.name}_output"
        output_config = self.create_output_config(
            self.config.resource_type,
            self.config.module_name,
            self.resource_manager,
        )
        self.output_base_model = create_base_model(
            output_base_model_name, output_config, RetrieverOutput
        )

        logger.info(
            f"Component {self.config.name} of type {self.component_type} created."
        )

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.model_fields}")
        logger.info(f"Output BaseModel: {self.output_base_model.model_fields}")
    
    def prepare_input(self, state: Dict[str, Any]):
        """
        temporary, to be replaced by generic component input preparation function
        :param state:
        :return:
        """
        input_data = self.input_base_model.model_validate(state)
        return input_data

    def retrieve_assets(
        self, query: RetrieverInput, resource: Resource
    ) -> List[RetrievedAsset]:
        """
        :param query: RetrieverInput for tag retrieval method
        :return: list of retrieved record
        """
        
        all_retrieved = []
        for record in resource.data:
            retrieved_record = {
                "asset_type": resource.type,
                "content": record,
                "relevance_score": 1,
            }
            retrieved_asset = RetrievedAsset.model_validate(retrieved_record)
            all_retrieved.append(retrieved_asset)
        logger.info(
            f"With input {query}, retrieved {len(all_retrieved)} records of {resource.type}."
        )
        return all_retrieved

    async def run(
            self, input_data: RetrieverInput = None, config: Dict = {}
    ) -> RetrieverOutput:
        """
        TODO: filter fields of retrieved asset to base model of component output
        :param query: RetrieverInput for tag retrieval method
        :return: output base model for retriever component
        """
        resource_type_module_combinations = list(
            itertools.product(
                self.config.resource_type, self.config.module_name
            )
        )
        component_output = {
            "component_name": self.config.name,
            "component_type": self.component_type,
        }
        retrieved_asset_all = []
        start_time = time.time()
        for resource_type, module_name in resource_type_module_combinations:
            resource = self.resource_manager.get_resource(
                resource_type, module_name
            )
            if resource is None:
                continue
            retrieved_asset = self.retrieve_assets(input_data, resource)

            retrieved_asset_all.extend(retrieved_asset)
            output_str = self.prepare_output_string(retrieved_asset, resource)
            if (
                "token_limit" in self.config.parameters
                and len(output_str.split())
                > self.config.parameters["token_limit"]
            ): # TODO: implement token counting
                logger.warning(
                    f"Output of {self.config.name} component is too long; trigger vector based retrieval"
                )
            # TODO: trigger vector retrieval
            component_output[
                self.get_state_name(resource_type, module_name)
            ] = output_str
        retrieve_time = time.time() - start_time

        component_output["metadata"] = {"time": retrieve_time}
        component_output["output_data"] = retrieved_asset_all

        return self.output_base_model.model_validate(component_output)


def test_all_retriever(use_case: str = "cdao_dia"):
    if use_case == "cdao_dia":
        config = yaml.safe_load(
            open("examples/cdao_dia/agent/config_graph_building.yml", "r")
        )
        my_resource_manager = ResourceManager(
            config["components"][3]["params"]["config"]
        )
        mock_state = {"rewritten_query": ""}
        component_config = config["components"][7]
    elif use_case == "ccb_risk":
        config = yaml.safe_load(
            open("examples/ccb_risk/config/config_ccb_risk.yml", "r")
        )
        my_resource_manager = ResourceManager(
            config["components"][4]["params"]["config"]
        )
        my_resource_manager.load_schema_embedding(
            data_file_path="examples/ccb_risk/data/schema_embedding.pkl"
        )
        mock_state = {
            "rewritten_query": "How have median cash buffers trended for Chase deposit customers since 2019?"
        }
        component_config = config["components"][7]
    retriever_node_config = component_config["params"]
    r_config = {"name": component_config["name"]}
    r_config.update(retriever_node_config["config"])
    r_input = retriever_node_config["input_config"]
    r_output = retriever_node_config["output_config"]

    all_retriever = AllRetriever(r_config, my_resource_manager, r_input, r_output)
    retriever_input = all_retriever.prepare_input(mock_state)
    my_retrieved_asset = asyncio.run(all_retriever.run(retriever_input))
    print("*" * 50)
    print(f"Component {all_retriever.config.name} of type {all_retriever.component_type} created.")
    print(f"Retrieved {len(my_retrieved_asset.output_data)} records in {my_retrieved_asset.metadata['time']:.2f} seconds")
    print("*"*50 + "\n" + my_retrieved_asset.dict()[r_output[0]]["name"])


if __name__ == "__main__":
    import asyncio
    import yaml
    test_all_retriever(use_case="ccb_risk")
    test_all_retriever(use_case="cdao_dia")