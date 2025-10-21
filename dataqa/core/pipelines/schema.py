from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from dataqa.errors import PipelineConfigError
from dataqa.pipelines.constants import PIPELINE_END, PIPELINE_START


class PipelineComponent(BaseModel):
    name: str
    type: str
    params: Dict[str, Any]
    input_source: Optional[Dict[str, str]] = None


class ParentGroup(BaseModel):
    parent: Union[str, List[str]]


class NodeEdge(BaseModel):
    name: str
    parent_groups: List[ParentGroup] = Field(
        description="""
            A list of parent groups.
            One parent group represents a group of nodes required together to trigger this nodess.
        """,
        default_factory=list,
    )


class Pipeline(BaseModel):
    name: str
    nodes: List[NodeEdge]

    @model_validator(mode="after")
    def valid_parent_groups(self):
        """
        Validate that every parent is a node in the pipeline

        raises ValueError
        """
        for node in self.nodes:
            for parent_group in node.parent_groups:
                if isinstance(parent_group.parent, str):
                    if (
                        not any(
                            [parent_group.parent == n.name for n in self.nodes]
                        )
                        and not parent_group.parent == PIPELINE_START
                    ):
                        raise PipelineConfigError(
                            f"Unknow node {parent_group.parent} used as the parent node of {node.name}"
                        )
                else:
                    for parent in parent_group.parent:
                        if (
                            not any([parent == n.name for n in self.nodes])
                            and not parent == PIPELINE_START
                        ):
                            raise PipelineConfigError(
                                f"Unknown node {parent} used as the parent node of {node.name}"
                            )
        return self


class PipelineConfig(BaseModel):
    components: List[PipelineComponent]
    pipelines: List[Pipeline]
    version: Optional[str] = None

    @model_validator(mode="after")
    def valid_node_name(self):
        """
        Validate that every node in pipeline is declared in components

        raises ValueError
        """
        for pipeline in self.pipelines:
            for node in pipeline.nodes:
                if (
                    not any(
                        [
                            node.name == component.name
                            for component in self.components
                        ]
                    )
                    and not node.name == PIPELINE_END
                ):
                    raise PipelineConfigError(
                        f"Unknown node {node.name} used in pipeline {pipeline.name}"
                    )
        return self

    def get_pipeline_definition(self, pipeline_name: str = None) -> Pipeline:
        """
        :param pipeline_name:
        :return:
        """

        if pipeline_name is None:
            if len(self.pipelines) == 0:
                raise PipelineConfigError(
                    "More than one pipelines specified in the config please specify the pipeline name"
                )
            else:
                return self.pipelines[0]

        pipelines = [
            pipeline
            for pipeline in self.pipelines
            if pipeline.name == pipeline_name
        ]

        if len(pipelines) == 1:
            return pipelines[0]

        if not pipelines:
            raise PipelineConfigError(
                f"No pipeline with name {pipeline_name} exists, please check your config"
            )

        if len(pipelines) != 1:
            raise PipelineConfigError(
                f"More than one pipeline with name {pipeline_name} present, please correct the config to provide a "
                f"unique pipeline name to every pipeline"
            )

    def get_component_by_name(self, component_name: str) -> PipelineComponent:
        """
        :param component_name:
        :return:
        """
        components = [
            component
            for component in self.components
            if component.name == component_name
        ]

        if len(components) == 1:
            return components[0]

        if not components:
            raise PipelineConfigError(
                f"No component with the name '{component_name}' found."
            )

        if len(components) > 1:
            raise PipelineConfigError(
                f"More than one components with name {component_name} present, please correct the config and provide a "
                f"unique component name to every component"
            )

    def get_component_definitions(self) -> Dict[str, Dict[str, Any]]:
        """

        :return:
        """
        component_defintions = {}
        for component in self.components:
            component_fields = {
                field: getattr(component, field)
                for field in component.model_fields.keys()
            }
            component_defintions[component.name] = component_fields

        return component_defintions
