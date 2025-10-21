import logging
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from dataqa.components.langgraph_conditional_edge.base_conditional_edge import (
    BaseConditionalEdge,
    BaseConditionalEdgeConfig,
    Condition,
)

logger = logging.getLogger(__name__)


class CategoricalVariableCondition(Condition):
    values: List[Any] = Field(description="allowed values")


class CategoricalVariableConditionEdgeConfig(BaseConditionalEdgeConfig):
    condition: List[CategoricalVariableCondition]


class CategoricalVariableConditionInput(BaseModel):
    variable: Any = Field(description="the variable to check in conditions")


class CategoricalVariableConditionEdge(BaseConditionalEdge):
    component_type = "CategoricalVariableConditionEdge"
    config_base_model = CategoricalVariableConditionEdgeConfig
    input_base_model = CategoricalVariableConditionInput
    config: CategoricalVariableConditionEdgeConfig

    def __init__(
        self,
        config: Union[CategoricalVariableConditionEdgeConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)

    def check_condition(
        self,
        condition: CategoricalVariableCondition,
        input_data: CategoricalVariableConditionInput,
    ) -> bool:
        for value in condition.values:
            if value == input_data.variable:
                return True
        return False

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.__fields__}")
        logger.info(f"Output BaseModel: {self.output_base_model.__fields__}")

    async def run(
        self, input_data: CategoricalVariableConditionInput, config: Dict
    ):
        for condition in self.config.condition:
            if self.check_condition(condition, input_data):
                logger.debug(
                    f"Value {input_data.variable} matches condition {condition.values}\nNext node is {condition.output}"
                )
                return condition.output
        logger.debug(
            f"No condition is matched by value {input_data.variable}.\nNext node is {self.config.default_output}"
        )
        return self.config.default_output
