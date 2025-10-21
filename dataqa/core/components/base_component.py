import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from dataqa.core.components.base_utils import get_field

logger = logging.getLogger(__name__)


class Variable(BaseModel):
    """Define a variable, can be used as the input or output for a tool."""

    name: str
    type: str
    description: Optional[str] = None
    optional: Optional[bool] = Field(
        description="If this variable is optional in the output", default=False
    )
    default: Optional[Any] = Field(
        description="If the variable has a default value.", default=None
    )


class OutputVariable(Variable):
    display: Optional[bool] = Field(
        description="If this variable appears in the output message to the orchestrator",
        default=True,
    )


class ComponentInput(BaseModel):
    """Base input for all components"""

    # Actual input models for the components are defined in the component classes
    # DISCUSS: component_name, component_type will be used for logging. we could also think about if we can use them as part of .run() method
    component_name: str = Field(description="Name of the target component")
    component_type: str = Field(description="Type of the target component")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the input"
    )
    # run_mode: langgraph


class ComponentOutput(BaseModel):
    """Base output for all components."""

    output_data: Any = Field(description="Output data of the component")
    # DISCUSS: component_name, component_type will be used for logging. we could also think about if we can use them as part of .run() method
    component_name: str = Field(
        description="Name of the component that produced this output"
    )
    component_type: str = Field(description="Type of the component")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the output (e.g.,  processing time, tokens)",
    )


class ComponentConfig(BaseModel):
    """Base configuration for all components."""

    name: str = Field(description="Name of the component instance")


class Component(ABC):
    """Abstract base class for all components"""

    is_component: bool = True
    input_mapping: Dict[str, str] = None
    output_mapping: Dict[str, str] = None

    def __init__(self, config: Union[ComponentConfig, Dict] = None, **kwargs):
        self.config = config
        if isinstance(config, Dict):
            self.config = self.config_base_model(**config)
        if not config:
            self.config = self.config_base_model(**kwargs)

    @property
    @abstractmethod
    def config_base_model(self) -> Type[BaseModel]:
        raise NotImplementedError

    @property
    @abstractmethod
    def component_type(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_base_model(self) -> Type[BaseModel]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_base_model(self) -> Type[BaseModel]:
        raise NotImplementedError

    @classmethod
    def memory_required(cls):
        return False

    @abstractmethod
    async def run(
        self, input_data: ComponentInput, config: RunnableConfig
    ) -> ComponentOutput:
        """Abstract method to execute the component's logic"""
        pass

    def display(self):
        pass

    def set_input_mapping(self, mapping):
        # validate
        fields = self.input_base_model.model_fields
        for field in mapping:
            if field not in fields:
                raise ValueError(
                    f"Field '{field}' is not defined in the input of {self.component_type}"
                )
        for field_name, field_info in fields.items():
            if field_info.is_required() and field_name not in mapping:
                raise ValueError(
                    f"Field '{field_name}' is required by the input model of {self.component_type}, but it was not provided in the input mapping."
                )

        self.input_mapping = mapping

    async def __call__(self, state, config: Optional[RunnableConfig] = {}):
        # build input data from state
        input_data = {
            field: get_field(state, mapped_field)
            for field, mapped_field in self.input_mapping.items()
        }

        input_data = self.input_base_model(**input_data)

        # run
        response = await self.run(input_data=input_data, config=config)

        # validate output and update state
        assert isinstance(response, self.output_base_model)

        if self.output_mapping:
            output = {}
            for k, v in self.output_mapping.items():
                if not hasattr(response, k):
                    warnings.warn(
                        f"Field '{k}' is missing in the output of {self.config.name}"
                    )
                else:
                    output[v] = getattr(response, k, None)
            return output
        else:
            return {f"{self.config.name}_output": response}
