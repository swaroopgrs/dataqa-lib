from typing import Callable, Dict, List, Tuple, Union

from langchain_core.tools import StructuredTool

from dataqa.core.memory import Memory
from dataqa.core.tools.analytics.tool_generator import DEFAULT_ANALYTICS_TOOLS
from dataqa.core.tools.plot.tool_generator import DEFAULT_PLOT_TOOLS
from dataqa.core.tools.utils import format_tool_description_with_indents


def get_tools_and_descriptions(
    memory: Memory,
    tool_names: Union[List[str], Dict[str, Callable]],
    all_tools_dict: Dict[str, Callable],
) -> Tuple[List[StructuredTool], str, str]:
    tools = []
    short_descriptions = []

    for name in tool_names:
        if name not in all_tools_dict:
            raise ValueError(f"Tool {name} is not defined.")
        tool, short_description, long_description = all_tools_dict[name](
            memory=memory
        )
        tools.append(tool)
        short_descriptions.append(short_description)

    names = [tool.name for tool in tools]

    short_description = format_tool_description_with_indents(
        names=names, descriptions=short_descriptions
    )
    long_description = format_tool_description_with_indents(
        names=names, descriptions=[tool.description for tool in tools]
    )

    return tools, short_description, long_description


def get_analytics_tools_and_descriptions(
    memory: Memory,
    tool_names: Union[List[str], Dict[str, Callable]] = DEFAULT_ANALYTICS_TOOLS,
) -> Tuple[List[StructuredTool], str, str]:
    return get_tools_and_descriptions(
        memory=memory,
        tool_names=tool_names,
        all_tools_dict=DEFAULT_ANALYTICS_TOOLS,
    )


def get_plot_tools_and_descriptions(
    memory: Memory,
    tool_names: Union[List[str], Dict[str, Callable]] = DEFAULT_PLOT_TOOLS,
) -> Tuple[List[StructuredTool], str, str]:
    return get_tools_and_descriptions(
        memory=memory, tool_names=tool_names, all_tools_dict=DEFAULT_PLOT_TOOLS
    )
