from typing import List

from dataqa.data_models.asset_models import Example, Rule, TableSchema
from dataqa.utils.schema_util import convert_table_schema_to_sql_str


def format_rules_for_prompt(rules: List[Rule]) -> str:
    """
    Takes a list of Rule objects and formats them into a single string
    by joining their instructions.
    """
    if not rules:
        return ""
    instructions_list = [rule.instructions for rule in rules]
    return "\n\n".join(instructions_list)


def format_examples_for_prompt(examples: List[Example]) -> str:
    """
    Takes a list of Example objects and formats them into a Q&A string
    suitable for few-shot prompting.
    """
    if not examples:
        return ""

    example_str_list = []
    for example in examples:
        content = example.example
        reasoning_str = (
            f"<reasoning>\n{content.reasoning}\n</reasoning>\n"
            if content.reasoning
            else ""
        )
        example_str = (
            f"Q: {content.question}\n"
            f"A: \n"
            f"{reasoning_str}"
            f"<sql>\n{content.code}\n</sql>"
        )
        example_str_list.append(example_str)

    return "\n\n".join(example_str_list)


def format_schema_for_prompt(tables: List[TableSchema]) -> str:
    """
    Takes a list of TableSchema objects and formats them into a single
    SQL DDL string.
    """
    if not tables:
        return ""

    schema_str_list = [
        convert_table_schema_to_sql_str(t.model_dump()) for t in tables
    ]
    return "\n".join(schema_str_list)