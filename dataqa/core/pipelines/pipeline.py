from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from dataqa.core.components.base_component import Component
from dataqa.core.errors import PipelineConfigError
from dataqa.core.pipelines.constants import (
    COMP_PREFIX,
    COMPONENT_MARKER,
    CONDITIONAL_EDGE_MARKER,
    FILE_PREFIX,
    INPUT_FROM_STATE,
    OUTPUT_TO_STATE,
    PIPELINE_END,
    PIPELINE_INPUT,
    PIPELINE_START,
    STATE_GRAPH_TYPE,
)
from dataqa.core.pipelines.schema import PipelineConfig
from dataqa.core.state import BasePipelineState
from dataqa.core.utils.utils import cls_from_str, load_file


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


def add_field(
    fields: Dict[str, Tuple], model: Type[BaseModel], schema: Dict[str, Dict]
):
    for output_name, field_name in schema["output_to_state"].items():
        if output_name not in model.model_fields:
            raise ValueError(
                f"Field {output_name} is not defined in the output of {schema['name']}."
            )
        field = model.model_fields[output_name]

        # check annotation, default, default_factory, metadata
        if field_name in fields:
            existing_annotation, existing_field = fields[field_name]
            if existing_annotation != field.annotation:
                raise ValueError(
                    f"Field {field_name} is defined multiple times with different annotations: {existing_annotation} vs {field.annotation}"
                )
            if existing_field.default != field.default:
                raise ValueError(
                    f"Field {field_name} is defined multiple times with different defaults: {existing_field.default} vs {field.default}"
                )
            if existing_field.default_factory != field.default_factory:
                raise ValueError(
                    f"Field {field_name} is defined multiple times with different default_factory: {existing_field.default_factory} vs {field.default_factory}"
                )
            if existing_field.metadata != field.metadata:
                raise ValueError(
                    f"Field {field_name} is defined multiple times with different metadata: {existing_field.metadata} vs {field.metadata}"
                )
        fields[field_name] = (field.annotation, field)


def build_graph_from_config(
    pipeline_schema: PipelineConfig, pipeline_name: Optional[str] = None
) -> Tuple[CompiledGraph, Type[BaseModel]]:
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
    for node_name, schema in component_definitions.items():
        component_instance = load_or_get_component(
            node_name, component_definitions, components
        )
        if getattr(component_instance, COMPONENT_MARKER, False) and not getattr(
            component_instance, CONDITIONAL_EDGE_MARKER, False
        ):
            add_field(
                fields=pipeline_state_fields,
                model=component_instance.output_base_model,
                schema=schema,
            )
    # Set default values for pipeline_state_fields
    for field_name, (annotation, field) in pipeline_state_fields.items():
        if hasattr(annotation, "__origin__") and annotation.__origin__ in (list, dict):
            if field.default_factory is None:
                pipeline_state_fields[field_name] = (
                    annotation,
                    Field(default_factory=annotation.__origin__, description=field.description),
                )
        elif field.default is None and field.default_factory is None:
            pipeline_state_fields[field_name] = (
                annotation,
                Field(default=None, description=field.description),
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
                for parent_group in node.parent_groups:
                    parent = parent_group.parent
                    if isinstance(parent, list) and len(parent) > 1:
                        raise PipelineConfigError(
                            f"{node.name} is an conditional edge. Each parent group could have only one parent node."
                        )
                    if isinstance(parent, list):
                        parent = parent[0]
                    graph_workflow.add_conditional_edges(
                        update_edge_node_name(parent),
                        component_instance.get_function(),
                    )
                # set input mapping
                input_mapping = component_definitions[node.name].get(
                    INPUT_FROM_STATE, {}
                )
                for field, mapped_field in input_mapping.items():
                    # replace "START" to "input"
                    names = mapped_field.split(".")
                    if names[0] == PIPELINE_START:
                        names[0] = PIPELINE_INPUT
                    # check if field exists in the input_base_model
                    if (
                        field
                        not in component_instance.input_base_model.model_fields
                    ):
                        raise ValueError(
                            f"Field {field} from {INPUT_FROM_STATE} does not exist in the input of {node.name}."
                        )
                    # check if this field exists in the state and if the type is consistent
                    current_model = pipeline_state_type
                    for name in names:
                        if name not in current_model.model_fields:
                            raise ValueError(
                                f"Field {mapped_field} from {INPUT_FROM_STATE} of node {node.name} does not exist in the state."
                            )
                        current_model = current_model.model_fields[name].annotation
                    if (
                        component_instance.input_base_model.model_fields[
                            field
                        ].annotation
                        != current_model
                    ):
                        raise ValueError(
                            f"Field {field} is required to be {component_instance.input_base_model.model_fields[field].annotation} as the input of {node.name}. But "
                            f"it is defined as {current_model} in the state."
                        )
                    input_mapping[field] = ".".join(names)
                component_instance.set_input_mapping(input_mapping)
                # set output mapping
                # output mapping has been verified when build the state.
                output_mapping = component_definitions[node.name].get(
                    OUTPUT_TO_STATE, {}
                )
                if not output_mapping and not getattr(
                    component_instance, CONDITIONAL_EDGE_MARKER, False
                ):
                    raise ValueError(
                        f"Component {node.name} has empty {OUTPUT_TO_STATE}."
                    )
                component_instance.output_mapping = output_mapping

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
    from pathlib import Path

    pipeline_config_path = Path(pipeline_path).resolve()
    config_dir = pipeline_config_path.parent
    with open(pipeline_config_path, "r") as f:
        pipeline_config_str = f.read().format(BASE_DIR=str(config_dir))
    pipeline_config = yaml.safe_load(pipeline_config_str)
    pipeline_schema = PipelineConfig(**pipeline_config)

    return build_graph_from_config(pipeline_schema, pipeline_name)
