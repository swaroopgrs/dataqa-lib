from typing import Dict, List, Union

import pandas as pd
from langchain_core.runnables import RunnableConfig

from dataqa.utils.dataframe_utils import df_to_markdown
from dataqa.utils.langgraph_utils import (
    CONFIGURABLE,
    DEFAULT_THREAD,
    MAX_TABLE_CHARACTERS,
    THREAD_ID,
)


class Memory:
    # TODO memory management
    # remove variables
    # summary
    dataframes: Dict[str, Dict[str, pd.DataFrame]]
    images: Dict[str, Dict[str, List[Union[str, pd.DataFrame]]]]

    def __init__(self):
        self.dataframes = {}
        self.images = {}

    def get_thread_id(self, config: RunnableConfig):
        return config.get(CONFIGURABLE, {}).get(THREAD_ID, DEFAULT_THREAD)

    def get_dataframes(self, config: RunnableConfig) -> Dict[str, pd.DataFrame]:
        thread_id = self.get_thread_id(config)
        if thread_id not in self.dataframes:
            self.dataframes[thread_id] = {}
        return self.dataframes[thread_id]

    def get_images(
        self, config: RunnableConfig
    ) -> Dict[str, List[Union[str, pd.DataFrame]]]:
        thread_id = self.get_thread_id(config)
        if thread_id not in self.images:
            self.images[thread_id] = {}
        return self.images[thread_id]

    def list_dataframes(self, config: RunnableConfig):
        return list(self.get_dataframes(config).keys())

    def get_dataframe(self, name: str, config: RunnableConfig):
        return self.get_dataframes(config).get(name)

    def put_dataframe(
        self, name: str, df: pd.DataFrame, config: RunnableConfig
    ):
        self.get_dataframes(config)[name] = df

    def get_image(self, name: str, config: RunnableConfig):
        return self.get_images(config).get(name)[0]

    def get_image_data(self, name: str, config: RunnableConfig):
        return self.get_images(config).get(name)[1]

    def put_image(
        self,
        name: str,
        img: List[Union[str, pd.DataFrame]],
        config: RunnableConfig,
    ):
        self.get_images(config)[name] = img

    def summarize_one_dataframe(self, df_name: str, df: pd.DataFrame):
        message = (
            f"  - dataframe_name: {df_name}\n"
            f"    size: {len(df)} rows, {len(df.columns)} columns\n"
        )
        sampled_rows = df_to_markdown(df.sample(n=min(5, len(df))).sort_index())
        if len(sampled_rows) < MAX_TABLE_CHARACTERS:
            return (
                message
                + "    Five sample rows:\n"
                + "\n".join([f"    {s}" for s in sampled_rows.split("\n")])
            )
        return message  # TODO better handle long tables.

    def summarize_dataframe(self, config: RunnableConfig):
        dataframes = self.get_dataframes(config)
        if not dataframes:
            return "You don't have access any dataframes yet."

        message = [
            f"You have access to the following {len(dataframes)} dataframes:"
        ]
        for k, v in dataframes.items():
            message.append(self.summarize_one_dataframe(k, v))
        return "\n\n".join(message)

    def summarize(self, name):
        pass


