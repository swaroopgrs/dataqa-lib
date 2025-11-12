import itertools
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from dataqa.core.components.base_component import (
    Component,
    ComponentConfig,
)
from dataqa.core.components.resource_manager.resource_manager import (
    ResourceManager,
)
from dataqa.core.data_models.asset_models import (
    Example,
    ResourceType,
    RetrievedAsset,
    Rule,
    TableSchema,
)
from dataqa.core.utils import asset_formatter
from dataqa.core.utils.data_model_util import create_base_model

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    TAG = "tag"
    VECTOR = "vector"
    HYBRID = "hybrid"
    ALL = "all"


class RetrieverInput(BaseModel):
    query: Any = Field(description="Query for retrieving the asset")


class RetrieverConfig(ComponentConfig):
    resource_types: List[ResourceType] = Field(
        description="List of resource types to retrieve: rule, schema, example"
    )
    module_names: List[str] = Field(
        description="List of module names to retrieve resources for."
    )
    retrieval_method: RetrievalMethod
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the retriever component.",
    )


class RetrieverOutput(BaseModel):
    output_data: List[RetrievedAsset] = Field(
        description="list of retrieved assets"
    )


class Retriever(Component, ABC):
    config: RetrieverConfig

    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self.retrieval_method = self.config.retrieval_method
        self.parameters = self.config.parameters

    @abstractmethod
    def retrieve_assets(
        self,
        query: Any,
        resources: List[Union[Rule, Example, TableSchema]],
        resource_type: ResourceType,
    ) -> List[RetrievedAsset]:
        pass

    async def run(
        self, input_data: RetrieverInput, config: Dict = {}
    ) -> RetrieverOutput:
        pass

    @staticmethod
    def prepare_output_string(
        retrieved_assets: List[RetrievedAsset], resource_type: ResourceType
    ) -> str:
        """
        Delegates formatting to the central AssetFormatter utility.
        """
        content_list = [asset.content for asset in retrieved_assets]

        if resource_type == ResourceType.Rule:
            return asset_formatter.format_rules_for_prompt(content_list)
        elif resource_type == ResourceType.Example:
            return asset_formatter.format_examples_for_prompt(content_list)
        elif resource_type == ResourceType.Schema:
            return asset_formatter.format_schema_for_prompt(content_list)

        return ""

    @staticmethod
    def create_output_config(
        resource_type_list: List[ResourceType],
        module_name_list: List[str],
    ) -> List[Dict]:
        output_config = []
        for resource_type, module_name in list(
            itertools.product(resource_type_list, module_name_list)
        ):
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
        if isinstance(config, Dict):
            retriever_config = RetrieverConfig.model_validate(config)
        else:
            retriever_config = config
        super().__init__(retriever_config)
        self.resource_manager = resource_manager

        output_base_model_name = f"{self.config.name}_output"
        output_config = self.create_output_config(
            self.config.resource_types,
            self.config.module_names,
        )
        self.output_base_model = create_base_model(
            output_base_model_name, output_config, RetrieverOutput
        )

        logger.info(
            f"Component {self.config.name} of type {self.component_type} initialized."
        )

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.model_fields}")
        logger.info(f"Output BaseModel: {self.output_base_model.model_fields}")

    def retrieve_assets(
        self,
        query: RetrieverInput,
        resources: List[Union[Rule, Example, TableSchema]],
        resource_type: ResourceType,
    ) -> List[RetrievedAsset]:
        """
        For AllRetriever, simply wrap all provided resources into RetrievedAsset objects.
        """
        all_retrieved = []
        for record in resources:
            retrieved_record = {
                "asset_type": resource_type.value,
                "content": record,
                "relevance_score": 1.0,
            }
            retrieved_asset = RetrievedAsset.model_validate(retrieved_record)
            all_retrieved.append(retrieved_asset)
        return all_retrieved

    async def run(
        self, input_data: RetrieverInput = None, config: Dict = {}
    ) -> RetrieverOutput:
        """
        Iterates through configured resource types and modules, retrieves all assets,
        and formats them into strings for the pipeline state.
        """
        resource_type_module_combinations = list(
            itertools.product(
                self.config.resource_types, self.config.module_names
            )
        )
        component_output = {}
        retrieved_asset_all = []

        for resource_type, module_name in resource_type_module_combinations:
            resources = self.resource_manager.get_resource(
                resource_type, module_name
            )

            state_name = self.get_state_name(resource_type, module_name)
            if not resources:
                component_output[state_name] = ""
                continue

            retrieved_assets = self.retrieve_assets(
                input_data, resources, resource_type
            )
            retrieved_asset_all.extend(retrieved_assets)

            output_str = self.prepare_output_string(
                retrieved_assets, resource_type
            )

            component_output[state_name] = output_str

        component_output["output_data"] = retrieved_asset_all

        return self.output_base_model.model_validate(component_output)
