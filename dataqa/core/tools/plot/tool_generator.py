import asyncio
from enum import Enum
from io import BytesIO
from typing import Annotated, Literal, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("agg")
import matplotlib.pyplot as plt
from langchain.tools import StructuredTool, tool
from langchain_core.runnables import RunnableConfig

from dataqa.core.memory import Memory
from dataqa.core.tools.utils import (
    no_dataframe_message,
)

lock = asyncio.Lock()


class PlotType(Enum):
    scatter = "scatter"
    bar = "bar"
    line = "line"
    pie = "pie"
    hist = "hist"
    box = "box"


def get_plot_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "Plot"
    # plot_engine = ""  # matplotlib, seaborn, plotly. Need tool config?
    short_description = "Tool to generate a plot with data in a dataframe, and save the binary string of the image with output image name.\nPlease name the output image with dataframe name and plot type"
    long_description = f"""
        {short_description}

        Parameter:
        - dataframe_name: the name of the dataframe to be used for plotting.
        - plot_type: Type of plot to produce. Supported types are:
                - 'scatter': Requires col_x and col_y.
                - 'bar': Requires col_x and col_y.
                - 'line': Requires col_x and col_y.
                - 'pie': Requires col_x for label and col_y for data.
                - 'hist': Requires only col_x.
                - 'box": Requires col_x and col_y.
        - col_x: the name of the column for x axis.
        - col_y: the name of the column for y axis.
        - output_image_name: output image name.

        Return:
        - A string as a message of completion or any exceptions.
"""

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def Plot(
        dataframe_name: Annotated[str, "Name of the dataframe to plot."],
        plot_type: Annotated[
            str,
            Literal["scatter", "bar", "line", "pie", "hist", "box"],
            "Plot type.",
        ],
        col_x: Annotated[str, "Column name for x-axis"] = None,
        col_y: Annotated[str, "Column name for y-axis"] = None,
        output_image_name: Annotated[str, "Name of the output image"] = None,
        config: Annotated[
            RunnableConfig, "Langchain RunnableConfiguration"
        ] = {},
    ) -> str:
        """
        Tool to generate a plot with data in a dataframe, and save the binary string of the image with output image name.
        Please name the output image with dataframe name and plot type

        Parameter:
        - dataframe_name: the name of the dataframe to be used for plotting.
        - plot_type: Type of plot to produce. Supported types are:
                - 'scatter': Requires col_x and col_y.
                - 'bar': Requires col_x and col_y.
                - 'line': Requires col_x and col_y.
                - 'pie': Requires col_x for label and col_y for data.
                - 'hist': Requires only col_x.
                - 'box": Requires col_x and col_y.
        - col_x: the name of the column for x axis.
        - col_y: the name of the column for y axis.
        - output_image_name: output image name.

        Return:
        - A string as a message of completion or any exceptions."""
        async with lock:
            try:
                try:
                    plot_type = PlotType(plot_type)
                except Exception:
                    raise ValueError(
                        f"Plot type {plot_type} not supported. Please choose from scatter, bar, line, pie, hist and box."
                    )

                df = memory.get_dataframe(dataframe_name, config=config)
                if df is None:
                    raise ValueError(no_dataframe_message(dataframe_name))
                df_plot = None

                match plot_type:
                    case PlotType.scatter:
                        # Scatter plot
                        sns.scatterplot(data=df, x=col_x, y=col_y)

                    case PlotType.bar:
                        # Bar plot
                        sns.barplot(data=df, x=col_x, y=col_y)
                        plt.xticks(rotation=45)
                        plt.tight_layout()

                    case PlotType.line:
                        # Line
                        sns.lineplot(data=df, x=col_x, y=col_y)

                    case PlotType.pie:
                        # Pie
                        plt.pie(x=df[col_y], labels=df[col_x])

                    case PlotType.hist:
                        # Histogram
                        if len(df[col_x]) < 2:
                            raise ValueError(
                                "Can NOT create histogram of data with only 1 record."
                            )
                        bins = np.histogram_bin_edges(df[col_x], bins=20)
                        counts, bin_edges = np.histogram(df[col_x], bins=bins)
                        df_plot = pd.DataFrame({"count_per_bin": counts})
                        sns.histplot(data=df, x=col_x, bins=bins)

                    case PlotType.box:
                        # Box plot
                        sns.boxplot(data=df, x=col_x, y=col_y)
                buffer = BytesIO()
                plt.savefig(buffer, format="png")
                binary_data = buffer.getvalue()

                if df_plot is None:
                    plot_columns = [col_x]
                    if col_y is not None:
                        plot_columns.append(col_y)
                    df_plot = df[plot_columns]
                memory.put_image(
                    output_image_name, [binary_data, df_plot], config=config
                )
                # Test async lock
                # plt.savefig(f"./temp/{output_image_name}.png")
                # await asyncio.sleep(30)
                plt.close("all")
                success_msg = f"Plot has been successfully generated, and image {output_image_name} saved."
                return f"{success_msg}\nSummary of plot data:\n{memory.summarize_one_dataframe(output_image_name, df_plot)}"
            except Exception as e:
                return f"Tool {name} failed with the following exception\n{repr(e)}"

    return Plot, short_description, long_description


DEFAULT_PLOT_TOOLS = {"Plot": get_plot_tool}
