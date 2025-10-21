import itertools
import logging
import time
from typing import Any, Dict, List

import yaml

from dataqa.components.resource_manager.resource_manager import (
    Resource,
    ResourceManager,
)
from dataqa.components.retriever.base_retriever import (
    Retriever,
    RetrieverConfig,
    RetrieverInput,
    RetrieverOutput,
)
from dataqa.data_models.asset_models import RetrievedAsset
from dataqa.utils.data_model_util import create_base_model

logger = logging.getLogger(__name__)


class TagRetriever(Retriever):
    component_type = "TagRetriever"
    config_base_model = RetrieverConfig
    input_base_model = "dynamically built"
    output_base_model = "dynamically built"

    def __init__(
        self,
        config: Dict,
        resource_manager: ResourceManager,
        input_config: List,
        output_config: List,
    ):
        """
        Create a new instance of the TagRetriever.

        Args:
           config (Dict): The configuration for the retriever.
           resource_manager (ResourceManager): The resource manager.
           input_config (List): The configuration for the input fields.
           output_config (List): The configuration for the output fields.

        Returns:
           TagRetriever: A new instance of the TagRetriever class.
        """

        tag_retriever_config = RetrieverConfig.model_validate(config)
        super().__init__(tag_retriever_config)
        self.resource_manager = resource_manager
        input_base_model_name = f"{self.config.name}_input"
        self.input_base_model = create_base_model(
            input_base_model_name, input_config
        )
        output_base_model_name = f"{self.config.name}_output"
        if output_config is None:
            output_config = self.create_output_config(
                self.config.resource_type,
                self.config.module_name,
                self.resource_manager,
            )
            self.output_field_name = None
        else:
            self.output_field_name = output_config[0]["name"]
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
        Retrieves assets from the resource based on the query.

        Args:
           query (RetrieverInput): The query to match against the resource.
           resource (Resource): The resource to retrieve assets from.

        Returns:
           list[RetrievedAsset]: A list of retrieved assets.
        """
        search_field = [r for r in self.input_base_model.model_fields]
        if isinstance(search_field, str):
            pass
        elif isinstance(search_field, list):
            if len(search_field) > 1:
                raise NotImplementedError(
                    f"Algorithm of multiple search fields for tag retriever is not implemented. Search field: {search_field}"
                )
            else:
                search_field = search_field[0]
        else:
            raise NotImplementedError(
                f"Algorithm of search fields of type {type(search_field)} for tag retriever is not implemented. Search field: {search_field}"
            )

        all_retrieved = []
        for record in resource.data:
            record_tag = getattr(record, search_field)
            input_tag = getattr(query, search_field)
            if self.validate(input_tag, record_tag):
                retrieved_record = {
                    "asset_type": resource.type,
                    "content": record,
                    "relevance_score": 1,
                }
                retrieved_asset = RetrievedAsset.model_validate(
                    retrieved_record
                )
                all_retrieved.append(retrieved_asset)
        logger.info(
            f"With input {query}, retrieved {len(all_retrieved)} records of {resource.type}."
        )
        return all_retrieved

    @staticmethod
    def validate(input_tag: list, asset_tag: list) -> bool:
        """
        :param input_tag: list of input tags
        :param asset_tag: list of tags of the asset record
        :return: boolean of whether the asset record should be selected
        """
        for conjunction in asset_tag:
            if not isinstance(conjunction, list):
                conjunction = [conjunction]
            f = True
            for predicate in conjunction:
                if predicate == "all":
                    return True
                # catalog has t, but predicate is ~t
                if predicate[0] == "~" and predicate[1:] in input_tag:
                    f = False
                    break
                # catalog doesn't have t, but predicate is t
                if predicate[0] != "~" and predicate not in input_tag:
                    f = False
                    break
            if f:
                return True
        return False

    async def run(
        self, input_data: RetrieverInput, config: Dict = {}
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
            if self.output_field_name is not None:
                component_output[self.output_field_name] = output_str
            else:
                component_output[
                    self.get_state_name(resource_type, module_name)
                ] = output_str
        retrieve_time = time.time() - start_time

        component_output["metadata"] = {"time": retrieve_time}
        component_output["output_data"] = retrieved_asset_all

        return self.output_base_model.model_validate(component_output)


if __name__ == "__main__":
    import asyncio

    config = yaml.safe_load(
        open("dataqa/examples/ccib_risk/config/config_retriever.yml", "r")
    )
    # my_kb = KnowledgeBase(config["components"][0]["params"]["config"])
    my_resource_manager = ResourceManager(
        config["components"][4]["params"]["config"]
    )

    mock_state = {"tags": ["trade"]}

    for component_config in config["components"][5:6]:
        retriever_node_config = component_config["params"]
        r_config = {"name": component_config["name"]}
        r_config.update(retriever_node_config["config"])
        r_input = retriever_node_config["input_config"]
        r_output = retriever_node_config["output_config"]

        tag_retriever = TagRetriever(
            r_config, my_resource_manager, r_input, r_output
        )
        tag_retriever_input = tag_retriever.prepare_input(mock_state)
        my_retrieved_asset = asyncio.run(tag_retriever.run(tag_retriever_input))
        print("*" * 50)
        print(
            f"Component {tag_retriever.config.name} of type {tag_retriever.component_type} created."
        )
        print(f"Retrieved {len(my_retrieved_asset.output_data)} records")
        print("Content:")
        for r in my_retrieved_asset.output_data:
            print(r.content.tags)
        print("*" * 50)
        print(
            f"Underlying string:\n{my_retrieved_asset.dict()[r_output[0]['name']]}"
        )
