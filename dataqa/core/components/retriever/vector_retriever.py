import itertools
import logging
import time
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import yaml
from numpy.linalg import norm

from dataqa.components.resource_manager.resource_manager import ResourceManager
from dataqa.components.retriever.base_retriever import (
    Retriever,
    RetrieverConfig,
    RetrieverInput,
    RetrieverOutput,
)
from dataqa.data_models.asset_models import Resource, RetrievedAsset
from dataqa.llm.openai import OpenAIEmbedding
from dataqa.utils.data_model_util import create_base_model

logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"


class VectorRetriever(Retriever):
    component_type = "VectorRetriever"
    config_base_model = RetrieverConfig
    input_base_model = "dynamically built"
    output_base_model = "dynamically built"

    def __init__(
        self,
        config: Dict,
        resource_manager: ResourceManager,
        input_config: List,
        output_config: List,
        embedding_model: OpenAIEmbedding,
    ):
        """
        Create a new instance of the VectorRetriever class.

        Args:
           config (Dict): The configuration for the retriever.
           resource_manager (ResourceManager): The resource manager.
           input_config (List): The configuration for the input fields.
           output_config (List): The configuration for the output fields.
           embedding_model (OpenAIEmbedding): The embedding model to use for vectorization.

        Returns:
           VectorRetriever: A new instance of the VectorRetriever class.
        """
        retriever_config = RetrieverConfig.model_validate(config)
        super().__init__(retriever_config)
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

        self.embedding_model = embedding_model

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

    async def retrieve_assets(
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
        all_retrieved = []

        arrays = [e.embedding_vector for e in resource.data]
        matrix = np.stack(arrays)

        query_embedding = await self.embedding_model(
            query.query, **self.embedding_model.config
        )
        query_embedding = np.array(query_embedding)

        if (
            self.config.parameters["distance_metric"]
            == DistanceMetric.DOT_PRODUCT.value
        ):
            scores = np.matmul(matrix, query_embedding)
        elif (
            self.config.parameters["distance_metric"]
            == DistanceMetric.COSINE.value
        ):
            norm_all = np.array([norm(f) for f in matrix])
            query_array = np.transpose(np.array(query_embedding))
            dot_prod = matrix.dot(query_array)
            dot_prod_flatten = dot_prod.flatten()
            scores = dot_prod_flatten / (norm(query_array) * norm_all)
        else:
            raise NotImplementedError
        top_k_idx = np.flip(
            scores.argsort()[-self.config.parameters["top_k"] :]
        )

        for i in top_k_idx:
            record = resource.data[i]
            retrieved_record = {
                "asset_type": resource.type,
                "content": record,
                "relevance_score": scores[i],
            }
            retrieved_asset = RetrievedAsset.model_validate(retrieved_record)
            all_retrieved.append(retrieved_asset)
        logger.info(
            f"With input {query}, retrieved {len(all_retrieved)} records of {resource.type}."
        )
        return all_retrieved

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
            retrieved_asset = await self.retrieve_assets(input_data, resource)
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
        open("examples/ccb_risk/config/config_ccb_risk.yml", "r")
    )
    my_resource_manager = ResourceManager(
        config["components"][4]["params"]["config"]
    )
    my_resource_manager.load_schema_embedding(
        data_file_path="examples/ccb_risk/data/schema_embedding.pkl"
    )
    from scripts.azure_token import get_az_token_using_cert

    api_key = get_az_token_using_cert()[0]

    embedding_model_config = {
        "azure_endpoint": "https://llmopenai-bi-us-east.openai.azure.com/openai/deployments/jpmc-ada-002-text-embedding/embeddings?api-version=2023-05-15",
        "openai_api_version": "2024-02-15",
        "api_key": api_key,
        "embedding_model_name": "text-embedding-ada-002",
    }
    embedding_model = OpenAIEmbedding()
    question = "How have median cash buffers trended for Chase deposit customers since 2021?"
    mock_state = {"query": question}
    component_config = config["components"][6:7]
    retriever_node_config = component_config["params"]
    r_config = {"name": component_config["name"]}
    r_config.update(retriever_node_config["config"])
    r_input = retriever_node_config["input_config"]
    r_output = retriever_node_config["output_config"]

    vector_retriever = VectorRetriever(
        r_config, my_resource_manager, r_input, r_output, embedding_model
    )
    vector_retriever_input = vector_retriever.prepare_input(mock_state)
    my_retrieved_asset = asyncio.run(
        vector_retriever.run(vector_retriever_input)
    )
    print("*" * 50)
    print(
        f"Component {vector_retriever.config.name} of type {vector_retriever.component_type} created."
    )
    print("*" * 50)
    print(f"Retrieved {len(my_retrieved_asset.output_data)} records")
    print("*" * 50)
    print(
        f"Underlying string:\n{my_retrieved_asset.dict()[r_output[0]['name']]}"
    )
