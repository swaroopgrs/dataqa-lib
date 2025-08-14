from typing import List


def no_dataframe_message(df_name):
    return f"Dataframe {df_name} is not found."


def format_tool_description_with_indents(
    names: List[str], descriptions: List[str]
) -> str:
    text = []
    for name, description in zip(names, descriptions):
        text.append(f"  - ToolName: {name}")
        text.append("    ToolDescription:")
        for line in description.split("\n"):
            text.append(f"      {line}")
    return "\n".join(text)