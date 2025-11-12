import logging

from pydantic import BaseModel

from dataqa.core.components.base_component import Component, ComponentConfig
from dataqa.core.state import PipelineOutput

logger = logging.getLogger(__name__)


class GatherOutputOutput(BaseModel):
    output: PipelineOutput = None


class GatherOutput(Component):
    config_base_model = ComponentConfig
    input_base_model = PipelineOutput
    output_base_model = GatherOutputOutput
    component_type = "GatherOutput"

    def display(self):
        logger.info("Gather PipelineOutput")

    async def run(self, input_data, config):
        return GatherOutputOutput(output=input_data)
