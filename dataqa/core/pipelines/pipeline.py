from typing import Any, Dict, List, Optional, Type, Union

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from pydantic import Field, create_model

from dataqa.errors import PipelineConfigError
from dataqa.pipelines.constants import (
    COMP_PREFIX,
    COMPONENT_MARKER,
    COMPONENT_OUTPUT_SUFFIX,
    CONDITIONAL_EDGE_MARKER,
    FILE_PREFIX,
    INPUT_SOURCE,
    PIPELINE_END,
    PIPELINE_INPUT,
    PIPELINE_START,
    STATE_GRAPH_TYPE,
)
from dataqa.pipelines.schema import PipelineConfig
from dataqa.state import BasePipelineState
from dataqa.utils.utils import cls_from_str, load_file


# TODO: Add support for loading files from resource manager
def load_or_get_component(
    component_name: str,
    component_definitions: Dict[str, Dict[str, Any]],
    components: Optional[Dict[str, Type]] = None,
):
    if component_name in components:
        return components[component_name]

    component_params = component_definitions[component_name].get("params", {})
    component_type = component_definitions[component_name]["type"]

    for key, value in component_params.items():
        if isinstance(value, str):
            if value.startswith(COMP_PREFIX):
                value_component_name = value.removeprefix(COMP_PREFIX)
                if value_component_name == component_name:
                    raise PipelineConfigError(
                        f"Component `{component_name}` references itself in its param `{key}`, please check the config"
                    )

                if value_component_name not in components:
                    load_or_get_component(
                        value_component_name, component_definitions, components
                    )

                component_params[key] = components[value_component_name]
            elif value.startswith(FILE_PREFIX):
                component_params[key] = load_file(
                    value.removeprefix(FILE_PREFIX)
                )

        if isinstance(value, dict):
            for val_key, val in value.items():
                if isinstance(val, str):
                    if val.startswith(COMP_PREFIX):
                        val_component_name = val.removeprefix(COMP_PREFIX)
                        if val_component_name == component_name:
                            raise PipelineConfigError(
                                f"Component `{component_name}` references itself in its param `{key}`, please check the config"
                            )

                        if val_component_name not in components:
                            load_or_get_component(
                                val_component_name,
                                component_definitions,
                                components,
                            )

                        value[val_key] = components[val_component_name]

                    elif val.startswith(FILE_PREFIX):
                        value[val_key] = load_file(
                            val.removeprefix(FILE_PREFIX)
                        )

    component_instance = cls_from_str(component_type)(**component_params)
    components[component_name] = component_instance

    return component_instance


def update_edge_node_name(node: Union[str, List[str]]) -> Union[str, List[str]]:
    def update_node_name(name):
        if name == PIPELINE_START:
            return START
        if name == PIPELINE_END:
            return END
        return name

    if isinstance(node, str):
        return update_node_name(node)

    return [update_node_name(name) for name in node]


def update_input_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    new_mapping = {}
    for field, mapped_field in mapping.items():
        names = mapped_field.split(".")
        if names[0] != PIPELINE_START:
            names[0] = f"{names[0]}{COMPONENT_OUTPUT_SUFFIX}"
        else:
            names[0] = PIPELINE_INPUT
        new_mapping[field] = ".".join(names)
    return new_mapping


def build_graph_from_config(
    pipeline_schema: PipelineConfig, pipeline_name: Optional[str] = None
) -> CompiledGraph:
    """

    :param pipeline_schema:
    :return:
    """

    # get pipeline definition
    pipeline_definition = pipeline_schema.get_pipeline_definition(pipeline_name)

    # get component definitions
    component_definitions = pipeline_schema.get_component_definitions()

    # Add some predefined fields to pipeline_state_fields
    components = {}
    pipeline_state_fields = {}

    # First pass to initialize all the components and add their output state to pipeline state
    for node_name in component_definitions.keys():
        component_instance = load_or_get_component(
            node_name, component_definitions, components
        )
        if getattr(component_instance, COMPONENT_MARKER, False) and not getattr(
            component_instance, CONDITIONAL_EDGE_MARKER, False
        ):
            pipeline_state_fields[f"{node_name}{COMPONENT_OUTPUT_SUFFIX}"] = (
                component_instance.output_base_model,
                Field(default=None, description=f"output of {node_name}"),
            )

    pipeline_state_type = create_model(
        STATE_GRAPH_TYPE, __base__=BasePipelineState, **pipeline_state_fields
    )
    graph_workflow = StateGraph(pipeline_state_type)

    nodes = [PIPELINE_END, PIPELINE_START]
    # add nodes to the graph

    for node in pipeline_definition.nodes:
        # add node
        if node.name not in [PIPELINE_START, PIPELINE_END]:
            component_instance = load_or_get_component(
                node.name, component_definitions, components
            )

            if not getattr(component_instance, CONDITIONAL_EDGE_MARKER, False):
                # Component
                # add node
                graph_workflow.add_node(node.name, component_instance)
                nodes.append(node.name)
                # add edges
                for parent_group in node.parent_groups:
                    graph_workflow.add_edge(
                        update_edge_node_name(parent_group.parent),
                        update_edge_node_name(node.name),
                    )
            else:
                # conditional edge, assert that conditional edge has EXACT one parent node
                if not len(node.parent_groups) == 1 or (
                    isinstance(node.parent_groups[0].parent, list)
                    and len(node.parent_groups[0].parent) != 1
                ):
                    raise PipelineConfigError(
                        f"{node.name} is an conditional edge. It requires exactly one parent node."
                    )
                parent = node.parent_groups[0].parent
                if isinstance(parent, list):
                    parent = parent[0]
                graph_workflow.add_conditional_edges(
                    update_edge_node_name(parent),
                    component_instance.get_function(),
                )

            # set input mapping
            if not component_definitions[node.name].get(INPUT_SOURCE, None):
                raise PipelineConfigError(
                    f"`{INPUT_SOURCE}` is required for {node.name} to define a node or an conditional edge"
                )
            mapping = {}
            for field, mapped_field in component_definitions[node.name][
                INPUT_SOURCE
            ].items():
                names = mapped_field.split(".")
                if names[0] != PIPELINE_START:
                    names[0] = f"{names[0]}{COMPONENT_OUTPUT_SUFFIX}"
                else:
                    names[0] = PIPELINE_INPUT
                mapping[field] = ".".join(names)
            component_instance.set_input_mapping(
                update_input_mapping(
                    component_definitions[node.name][INPUT_SOURCE]
                )
            )

        elif node.name == PIPELINE_END:
            for parent_group in node.parent_groups:
                graph_workflow.add_edge(
                    update_edge_node_name(parent_group.parent),
                    update_edge_node_name(node.name),
                )

    compiled_graph = graph_workflow.compile(checkpointer=MemorySaver())
    return compiled_graph, pipeline_state_type


def build_graph_from_yaml(
    pipeline_path: str, pipeline_name: Optional[str] = None
):
    pipeline_config = yaml.safe_load(open(pipeline_path))
    pipeline_schema = PipelineConfig(**pipeline_config)

    return build_graph_from_config(pipeline_schema, pipeline_name)

