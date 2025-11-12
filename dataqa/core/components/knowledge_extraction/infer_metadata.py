from typing import List, Literal

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from dataqa.core.llm.openai import AzureOpenAI
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    TOKEN,
)
from dataqa.core.utils.prompt_utils import build_prompt, prompt_type


class MetaData(BaseModel):
    """Metadata for the database"""

    level: Literal["table", "column"] = Field(
        description="level of the metadata"
    )
    name: str = Field(description="name of the column or table")
    description: str = Field(description="description of the column or table")


class MetaDataList(BaseModel):
    """Metadata for the database"""

    metadata: List[MetaData] = Field(
        default_factory=list, description="A list of metadata for the database"
    )


meta_inference_prompt = """As an AI assistant, you are given information (schema and/or sample data) of a SQL table.
Your task is to generate description of the table, and infer how data is structured in the table.

Here are some tasks you can perform:
1. Generate description of the table. In the description, you should also provide
  - how data is structured in the table
  - what each row represents
  - what is the relationship between each row
  - what is the relationship between rows
  - if there is any relationship between columns in the table, such as hierarchical relationship, derived relationship, etc.
2. Generate description of each column in the table. In the description, you should also provide
  - what each column represents
  - what is the function type of each column. Here are list of function types:
        - IDENTIFIER
        - CATEGORICAL VALUE (such as gender, country, company name, etc.)
        - NUMERIC
        - TEXT
        - DATETIME
        - BOOLEAN

Table information:
Table name: {table_name}
Table shape: {table_shape} (rows, columns)

Sample rows of the table:
{sample_rows}

Column information:
{column_info}

Please generate metadata of the table, and columns in the following format:
{{"level": "", "name": "", "description": ""}}
Here level should be one of "table" or "column"
"""


class MetaInference:
    name = "meta_inference"
    num_retries: int = 5

    def __init__(self, llm: AzureOpenAI, prompt: prompt_type):
        self.prompt = build_prompt(prompt)
        self.llm = llm

    async def __call__(
        self,
        table_name: str,
        table_shape: str,
        sample_rows: str,
        column_info: str,
        config: RunnableConfig,
    ):
        messages = self.prompt.invoke(
            dict(
                table_name=table_name,
                table_shape=table_shape,
                sample_rows=sample_rows,
                column_info=column_info,
            )
        )
        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        token = config.get(CONFIGURABLE, {}).get(TOKEN, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")

        responses = []
        for _ in range(self.num_retries):
            response = await self.llm.ainvoke(
                messages=messages,
                api_key=api_key,
                token=token,
                base_url=base_url,
                from_component=self.name,
                with_structured_output=MetaDataList,
            )
            responses.append(response)
            metadata = response.generation
            if isinstance(metadata, MetaDataList):
                break
        if not isinstance(metadata, MetaDataList):
            raise Exception("Failed to infer metadata.")
        return dict(metadata=metadata, llm_output=responses)
