from typing import Annotated, List, Literal, Tuple, Union

import pandas as pd
from langchain.tools import StructuredTool, tool
from langchain_core.runnables import RunnableConfig

from dataqa.memory import Memory
from dataqa.tools.utils import no_dataframe_message

valid_agg_funcs = [
    "sum",
    "mean",
    "max",
    "min",
    "count",
    "std",
    "var",
    "first",
    "last",
    "median",
    "prod",
    "nunique",
]


def get_df_tool_message(memory: Memory, df_name: str, df: pd.DataFrame) -> str:
    msg = "Here is the summary of the output dataframe: \n"
    if df.empty:
        msg = f"The output dataframe {df_name} is empty."
    else:
        msg += memory.summarize_one_dataframe(df_name, df)
        msg += "\nNote: The summary may only include sampled rows and/or columns of the dataframe."
    return msg


def get_correlation_matrix_tool(
    memory: Memory,
) -> Tuple[StructuredTool, str, str]:
    name = "CalculateCorrelationMatrix"
    short_description = "Compute pairwise correlation of columns for dataframe called `dataframe_name`, excluding NA/null values, save the correlation matrix as a new dataframe called `output_df_name`."
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    The name of the dataframe to calculate correlation.
output_df_name : str
    The name of the correlation matrix as a dataframe.
method : ['pearson', 'kendall', 'spearman'], default 'pearson'
    Method of correlation:
    * pearson : standard correlation coefficient
    * kendall : Kendall Tau correlation coefficient
    * spearman : Spearman rank correlation
min_periods : int, optional
    Minimum number of observations required per pair of columns
    to have a valid result. Currently only available for Pearson and Spearman correlation.
numeric_only : bool, default False
    Include only `float`, `int` or `boolean` data.

Returns
-------
Tool calling response : str
    - If successful, return a message saying that "The correlation matrix of dataframe `dataframe_name` has been calculated and saved in a new dataframe `output_df_name`."
    - If failed, return a message of the runtime exception.

Usage
-----
``IMPORTANT``: Before calling this tool, make sure that the input dataframe is in a good shape, that is:
    - Each column represents one object and we want to calculate the correlation between each pair of objects / columns.
    - Each row represents one feature. One object is described by its feature vector.
If needed, call transformation tool before calling this tool, such as PivotTable, GroupBy.

Examples
--------
Assume that we have a dataframe called "df_abc" with 5 rows and 3 columms A, B, C.

>>> print(df_abc)
            A         B         C
0  0.655982  0.990371  0.431369
1  0.093596  0.565008  0.873763
2  0.379816  0.965121  0.792393
3  0.479515  0.820517  0.055805
4  0.433931  0.845164  0.734673

Calculate the correlation matrix of df_abc in a dataframe df_abc_corr

>>> CalculateCorrelationMatrix(
...     dataframe_name="df_abc", output_df_name="df_abc_corr"
... )
>>> print(df_abc_corr)
        A         B         C
A  1.000000  0.861468 -0.613955
B  0.861468  1.000000 -0.288519
C -0.613955 -0.288519  1.000000
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def CalculateCorrelationMatrix(
        dataframe_name: Annotated[
            str, "Name of the dataframe to calculate correlation."
        ],
        output_df_name: Annotated[
            str,
            "Name of the output dataframe with calculated correlation matrix.",
        ],
        method: Annotated[
            Literal["pearson", "kendall", "spearman"],
            "Method used to calculate correlation.",
        ] = "pearson",
        min_periods: Annotated[
            int, "Minimum number of observations required per pair of columns"
        ] = 1,
        numeric_only: Annotated[
            bool, "Include only `float`, `int` or `boolean` data"
        ] = False,
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            df = memory.get_dataframe(dataframe_name, config=config)
            if df is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            out_df = df.T.corr(
                method=method,
                min_periods=min_periods,
                numeric_only=numeric_only,
            )
            memory.put_dataframe(output_df_name, out_df, config=config)
            success_msg = f"Dataframe {output_df_name} has been successfully generated as the correlation matrix of {dataframe_name}."
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, out_df)}"

        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return CalculateCorrelationMatrix, short_description, long_description


def get_n_largest_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "nLargest"
    short_description = """Return the first `n` rows with the largest values in `columns`, in descending order.\nThe columns that are not specified are returned as well, but not used for ordering."""
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    The name of the dataframe to get n-largest rows.
output_df_name : str
    The name of n-largest rows as a dataframe.
n : int
    Number of rows to return.
columns : column name or list of column names
    Column label(s) to order by.
keep : ['first', 'last', 'all'], default 'first'
    Where there are duplicate values:
    - ``first`` : prioritize the first occurrence(s)
    - ``last`` : prioritize the last occurrence(s)
    - ``all`` : keep all the ties of the smallest item even if it means selecting more than ``n`` items.

Returns
-------
Tool calling response: str
- If successful, return a message saying that "N-largest rows of dataframe `dataframe_name` has been calculated and saved in a new dataframe `output_df_name`."
- If failed, return a message of the runtime exception.

Examples
--------
Assume that we have a dataframe called "df_country"

>>> print(df_country)
            population      GDP alpha-2
Italy       59000000  1937894      IT
Malta         434000    12011      MT
Maldives      434000     4520      MV
Iceland       337000    17036      IS

Select two countries with the largest population

>>> nLargest(
...     dataframe_name="df_country",
...     output_df_name="df_top_2_population",
...     n=2,
...     columns=["population"],
... )
>>> print(df_top_2_population)
        population      GDP alpha-2
Italy     59000000  1937894      IT
Malta       434000    12011      MT

When using ``keep='last'``, ties are resolved in reverse order:

>>> nLargest(
...     dataframe_name="df_country",
...     output_df_name="df_top_2_population",
...     n=2,
...     columns=["population"],
...     keep="last",
... )
>>> print(df_top_2_population)
            population      GDP alpha-2
Italy       59000000  1937894      IT
Maldives      434000     4520      MV

When using ``keep='all'``, the number of element kept can go beyond ``n``
if there are duplicate values for the largest element, all the
ties are kept:
>>> nLargest(
...     dataframe_name="df_country",
...     output_df_name="df_top_2_population",
...     n=2,
...     columns=["population"],
...     keep="all",
... )
>>> print(df_top_2_population)
            population      GDP alpha-2
Italy       59000000  1937894      IT
Malta         434000    12011      MT
Maldives      434000     4520      MV
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def nLargest(
        dataframe_name: Annotated[str, "Dataframe to get n-largest rows from."],
        output_df_name: Annotated[
            str, "Name of n-largest rows as a dataframe."
        ],
        n: Annotated[int, "Number of rows to return."],
        columns: Annotated[
            Union[str, List[str]], "Column label(s) to order by."
        ],
        keep: Annotated[
            Literal["first", "last", "all"],
            "Which one to keep when there are duplicate values.",
        ] = "first",
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            df = memory.get_dataframe(dataframe_name, config=config)
            if df is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            out_df = df.nlargest(n=n, columns=columns, keep=keep)
            memory.put_dataframe(output_df_name, out_df, config=config)
            success_msg = f"Top {n} largest rows of dataframe {dataframe_name} has been calculated and saved in a new dataframe {output_df_name}."
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, out_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return nLargest, short_description, long_description


def get_n_smallest_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "nSmallest"
    short_description = "Return the first `n` rows with the smallest values in `columns`, in ascending order. The columns that are not specified are returned as well, but not used for ordering."
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    The name of the dataframe to get n-smallest rows.
output_df_name : str
    The name of n-smallest rows as a dataframe.
n : int
    Number of rows to return.
columns : column name or list of column names
    Column label(s) to order by.
keep : ['first', 'last', 'all'], default 'first'
    Where there are duplicate values:
    - ``first`` : prioritize the first occurrence(s)
    - ``last`` : prioritize the last occurrence(s)
    - ``all`` : keep all the ties of the smallest item even if it means selecting more than ``n`` items.

Returns
-------
Tool calling response: str
- If successful, return a message saying that "N-smallest rows of dataframe `dataframe_name` has been calculated and saved in a new dataframe `output_df_name`."
- If failed, return a message of the runtime exception.

Examples
--------
Assume that we have a dataframe called "df_country"

>>> print(df_country)
            population      GDP alpha-2
Italy       59000000  1937894      IT
Malta         434000    12011      MT
Maldives      434000     4520      MV
Iceland       337000    17036      IS

Select two countries with the smallest population

>>> nSmallest(
...     dataframe_name="df_country",
...     output_df_name="df_top_2_population",
...     n=2,
...     columns=["population"],
... )
>>> print(df_top_2_population)
        population      GDP alpha-2
Iceland     337000    17036      IS
Malta       434000    12011      MT

When using ``keep='last'``, ties are resolved in reverse order:

>>> nSmallest(
...     dataframe_name="df_country",
...     output_df_name="df_top_2_population",
...     n=2,
...     columns=["population"],
...     keep="last",
... )
>>> print(df_top_2_population)
            population      GDP alpha-2
Iceland       337000    17036      IS
Maldives      434000     4520      MV

When using ``keep='all'``, the number of element kept can go beyond ``n``
if there are duplicate values for the largest element, all the
ties are kept:
>>> nSmallest(
...     dataframe_name="df_country",
...     output_df_name="df_top_2_population",
...     n=2,
...     columns=["population"],
...     keep="all",
... )
>>> print(df_top_2_population)
            population      GDP alpha-2
Iceland       337000    17036      IS
Malta         434000    12011      MT
Maldives      434000     4520      MV
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def nSmallest(
        dataframe_name: Annotated[
            str, "Name of the dataframe to get n-smallest rows."
        ],
        output_df_name: Annotated[
            str, "Name of n-smallest rows as a dataframe."
        ],
        n: Annotated[int, "Number of rows to return."],
        columns: Annotated[
            Union[str, List[str]], "Column label(s) to order by."
        ],
        keep: Annotated[
            Literal["first", "last", "all"],
            "Which one to keep when there are duplicate values.",
        ] = "first",
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            df = memory.get_dataframe(dataframe_name, config=config)
            if df is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            out_df = df.nsmallest(n=n, columns=columns, keep=keep)
            memory.put_dataframe(output_df_name, out_df, config=config)
            success_msg = f"Top {n} smallest rows of dataframe {dataframe_name} has been calculated and saved in a new dataframe {output_df_name}."
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, out_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return nSmallest, short_description, long_description


def get_sort_value_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "SortValue"
    short_description = """Sort by the values along either axis."""
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    The name of the dataframe to sort.
output_df_name : str
    The name of the sorted dataframe.
by : str or list of str
    Name or list of names to sort by.
    - if `axis` is 0 or `'index'` then `by` may contain index levels and/or column labels.
    - if `axis` is 1 or `'columns'` then `by` may contain column levels and/or index labels.
axis : "[0 or 'index', 1 or 'columns']", default 0
        Axis to be sorted.
ascending : bool or list of bool, default True
    Sort ascending vs. descending. Specify list for multiple sort orders.  If this is a list of bools, must match the length of the by.

Returns
-------
Tool calling response: str
- If successful, return a message saying that "The sorted dataframe `dataframe_name` has been created and saved as a new dataframe `output_df_name`."
- If failed, return a message of the runtime exception.

Examples
--------
>>> df
    col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     9     9    c
3  NaN     8     4    D
4    D     7     2    e
5    C     4     3    F

Sort by col1

>>> SortValue(dataframe_name="df", output_dfd_name="df_sort", by=["col1"])
>>> print(df_sort)
    col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     9     9    c
5    C     4     3    F
4    D     7     2    e
3  NaN     8     4    D
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def SortValue(
        dataframe_name: Annotated[str, "Name of the dataframe to sort."],
        output_df_name: Annotated[str, "Name of the sorted dataframe."],
        by: Annotated[
            Union[str, List[str]], "Name or list of names to sort by."
        ],
        axis: Annotated[
            Union[int, Literal["index", "columns", "rows"]], "Axis to be sorted"
        ] = 0,
        ascending: Annotated[
            bool | list[bool] | tuple[bool, ...],
            "Sort ascending vs. descending",
        ] = True,
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            df = memory.get_dataframe(dataframe_name, config=config)
            if df is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            out_df = df.sort_values(by=by, axis=axis, ascending=ascending)
            memory.put_dataframe(output_df_name, out_df, config=config)
            success_msg = f"The sorted dataframe {dataframe_name} has been created and saved as a new dataframe {output_df_name}."
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, out_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return SortValue, short_description, long_description


def get_aggregrate_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "ColumnAggregation"

    short_description = "Tool to aggregate specified columns."
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    Name of the dataframe to use for column aggregation.
output_df_name : str
    Name of the new dataframe to create for the result
agg_columns : List[str]
    List of columns to aggregate
agg_funcs : List[str]
    List of aggregation functions to apply. Allowed operations are: 'sum', 'mean', 'max', 'min', 'count', 'std', 'var', 'first', 'last', 'median', 'prod', 'nunique'.
output_column_names : List[str]
    List of new names for the aggregated columns to avoid conflicts.

Returns
-------
Tool calling response: str
- If successful, return a message saying that "Aggregated dataframe created" and showing the indices of the new dataframe.
- If failed, return a message of the runtime exception.

Example:
------
>>> df
    A	B	C
0	1	2	3
1	4	5	6
2	7	8	9

Aggregate column A using max and aggregate column B using min.

>>> ColumnAggregation(
...     dataframe_name='df',
...     output_df_name='df_agg',
...     agg_columns=['A', 'B'],
...     agg_funcs=['max', 'min'],
...     output_column_names=['max_A', 'min_B']
)
>>> print(df_agg)
        A   B
max	 7.0   NaN
min	 NaN   2.0
"""

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def ColumnAggregation(
        dataframe_name: Annotated[
            str, "Name of the dataframe to use for column aggregation."
        ],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result."
        ],
        agg_columns: Annotated[list, "List of columns to aggregate."],
        agg_funcs: Annotated[
            list,
            "List of aggregation functions to apply for each column. The length of agg_funcs should be equal to agg_columns.",
        ],
        output_column_names: Annotated[
            list,
            "List of new names for the aggregated columns to avoid conflicts. If specified, the length of output_column_names should be equal to agg_columns.",
        ] = None,
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        """
        TODO: support agg_functions as list of list of agg functions.
        """
        try:
            dataframe = memory.get_dataframe(dataframe_name, config=config)
            if dataframe is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            if (
                not isinstance(agg_columns, list)
                or not isinstance(agg_funcs, list)
                or (
                    output_column_names is not None
                    and not isinstance(output_column_names, list)
                )
            ):
                raise ValueError(
                    "agg_columns, agg_funcs and output_column_names must be lists."
                )

            if len(agg_columns) != len(agg_funcs):
                raise ValueError(
                    "The length of agg_columns and agg_funcs must be the same."
                )

            if output_column_names and len(output_column_names) != len(
                agg_columns
            ):
                raise ValueError(
                    "The length of agg_columns and output_column_names must be the same."
                )

            agg_dict = {}
            for col, func in zip(agg_columns, agg_funcs):
                if col not in dataframe.columns:
                    raise ValueError(
                        f"Column {col} does NOT exist in dataframe {dataframe_name}."
                    )
                if func not in valid_agg_funcs:
                    raise ValueError(
                        f"Invalid aggregation function '{func}'. Valid functions are: {', '.join(valid_agg_funcs)}"
                    )
                if col not in agg_dict:
                    agg_dict[col] = []
                if func not in agg_dict[col]:
                    agg_dict[col].append(func)

            new_df = dataframe.aggregate(agg_dict)
            if isinstance(new_df, pd.Series):
                new_df = new_df.to_frame()
            elif not isinstance(new_df, pd.DataFrame):
                raise ValueError(
                    f"Failed to generate a new dataframe by calling column aggregation, the type of output is in the type of {type(new_df)}."
                )

            if output_column_names:
                if len(output_column_names) != len(new_df.columns):
                    raise ValueError(
                        "The length of output_column_names must match the number of resulting columns."
                    )
                new_df.columns = output_column_names

            memory.put_dataframe(output_df_name, new_df, config)

            success_msg = f"Aggregated dataframe created and stored as {output_df_name}. The new dataframe has the following indices: {new_df.index.to_list()}"
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, new_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return ColumnAggregation, short_description, long_description


def get_groupby_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "GroupBy"
    short_description = "Tool to perform groupby operation on a dataframe and aggregate specified columns."
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    Name of the dataframe to use for the groupby operation
output_df_name : str
    Name of the new dataframe to create for the result
groupby_columns : List[str]
    List of columns to group by
agg_columns : List[str]
    List of columns to aggregate
agg_funcs : List[str]
    List of aggregation functions to apply. Allowed operations are: 'sum', 'mean', 'max', 'min', 'count', 'std', 'var', 'first', 'last', 'median', 'prod', 'nunique'
output_column_names : List[str] | None
    List of new names for the aggregated columns to avoid conflicts. Default to None for using the original column names.

Returns
-------
A string indicating the result of the groupby operation, including the names of the aggregated columns.
If failed, return the runtime exception.
"""

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def GroupBy(
        dataframe_name: Annotated[
            str, "Name of the dataframe to use for the groupby operation."
        ],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result."
        ],
        groupby_columns: Annotated[list, "List of columns to group by."],
        agg_columns: Annotated[list, "List of columns to aggregate."],
        agg_funcs: Annotated[list, "List of aggregation functions to apply"],
        output_column_names: Annotated[
            list,
            "List of new names for the aggregated columns to avoid conflicts",
        ] = None,
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            dataframe = memory.get_dataframe(dataframe_name, config=config)
            if dataframe is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            if (
                not isinstance(groupby_columns, list)
                or not isinstance(agg_columns, list)
                or not isinstance(agg_funcs, list)
            ):
                raise ValueError(
                    "groupby_columns, agg_columns, and agg_funcs must be lists."
                )

            if len(agg_columns) != len(agg_funcs):
                raise ValueError(
                    "The length of agg_columns and agg_funcs must be the same."
                )

            for func in agg_funcs:
                if func not in valid_agg_funcs:
                    raise ValueError(
                        f"Invalid aggregation function '{func}'. Valid functions are: {', '.join(valid_agg_funcs)}"
                    )

            # Create a dictionary for aggregation
            agg_dict = {}
            for col, func in zip(agg_columns, agg_funcs):
                if col not in agg_dict:
                    agg_dict[col] = []
                agg_dict[col].append(func)

            grouped_df = dataframe.groupby(groupby_columns).agg(agg_dict)

            # Flatten the MultiIndex columns
            grouped_df.columns = [
                f"{col}_{func}"
                for col, funcs in agg_dict.items()
                for func in funcs
            ]

            # Rename columns to avoid conflicts
            if output_column_names:
                if len(output_column_names) != len(grouped_df.columns):
                    raise ValueError(
                        "The length of output_column_names must match the number of resulting columns."
                    )
                grouped_df.columns = output_column_names

            # Reset index without inserting the index as a column
            grouped_df = grouped_df.reset_index()

            if output_df_name:
                memory.put_dataframe(output_df_name, grouped_df, config=config)

            success_msg = (
                f"Grouped dataframe created and stored as '{output_df_name}'. Aggregated columns: {', '.join(grouped_df.columns)}"
                if output_df_name
                else f"{grouped_df.to_string()}\nAggregated columns: {', '.join(grouped_df.columns)}"
            )
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, grouped_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return GroupBy, short_description, long_description


def get_pivot_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "PivotTable"
    short_description = """Reshapes a dataframe into a pivot table to organize data for effective analysis.\nUse this tool when the dataframe's structure needs to be transformed for better analysis and visualization.\nPivoting is essential for converting row-based data into a more structured, column-based format."""
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    Name of the dataframe to pivot.
output_df_name : str
    Name of the new dataframe to create for the result
index : List[str] | None
    Column(s) to use as the pivot table index (rows).
columns : List[str] | None
    Column(s) to use as the pivot table column headers.
values : List[str] | None
    Column(s) to aggregate. For count operations, use a column different from those in 'columns' or set to None.
aggfunc : List[str]
    Aggregation function ('mean', 'sum', 'count', 'min', 'max', etc.).
add_totals : bool
    Whether to add row and column totals to the pivot table.

Returns
-------
A string indicating the pivot table creation result.
If failed, return the runtime exception.
"""

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def PivotTable(
        dataframe_name: Annotated[str, "Name of the dataframe to pivot."],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result."
        ],
        index: Annotated[
            list, "Column(s) to use as the pivot table index (rows)."
        ] = None,
        columns: Annotated[
            list, "Column(s) to use as the pivot table column headers."
        ] = None,
        values: Annotated[
            list,
            "Column(s) to aggregate. For count operations, use a column different from those in 'columns' or set to None.",
        ] = None,
        aggfunc: Annotated[
            str,
            "Aggregation function ('mean', 'sum', 'count', 'min', 'max', etc.)",
        ] = "mean",
        add_totals: Annotated[
            bool, "Whether to add row and column totals to the pivot table."
        ] = False,
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            dataframe = memory.get_dataframe(dataframe_name, config=config)

            if dataframe is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            # Validate and normalize parameters
            if not isinstance(index, list):
                index = [index]

            if columns is not None and not isinstance(columns, list):
                columns = [columns]

            if values is not None and not isinstance(values, list):
                values = [values]

            # Validate column existence
            all_columns = set(dataframe.columns)
            for col in index:
                if col not in all_columns:
                    raise ValueError(
                        f"Error: Index column '{col}' not found in the dataframe."
                    )

            if columns:
                for col in columns:
                    if col not in all_columns:
                        raise ValueError(
                            f"Error: Column '{col}' not found in the dataframe."
                        )

            if values:
                for col in values:
                    if col not in all_columns:
                        raise ValueError(
                            f"Error: Value column '{col}' not found in the dataframe."
                        )

            # Special handling for count operations
            if aggfunc.lower() == "count":
                # Check for the problematic case where values overlap with columns
                if values and columns:
                    values_set = set(values)
                    columns_set = set(columns)

                    if values_set.intersection(columns_set):
                        # Use crosstab for more reliable counting
                        index_data = [dataframe[col] for col in index]
                        col_data = [dataframe[col] for col in columns]

                        pivot_df = pd.crosstab(
                            index=index_data
                            if len(index) > 1
                            else index_data[0],
                            columns=col_data
                            if len(columns) > 1
                            else col_data[0],
                        )

                        # Set appropriate names
                        pivot_df.index.names = index
                        pivot_df.columns.names = columns
                    else:
                        # No overlap, use regular pivot_table
                        pivot_df = pd.pivot_table(
                            dataframe,
                            index=index,
                            columns=columns,
                            values=values,
                            aggfunc="count",
                        )
                else:
                    # If no values specified or no columns specified, use size
                    pivot_df = pd.pivot_table(
                        dataframe,
                        index=index,
                        columns=columns,
                        values=values if values else None,
                        aggfunc="size" if not values else "count",
                    )
            else:
                # For other aggregation functions
                pivot_df = pd.pivot_table(
                    dataframe,
                    index=index,
                    columns=columns,
                    values=values,
                    aggfunc=aggfunc,
                )

            # Reset index for better usability in subsequent operations
            pivot_df = pivot_df.reset_index()

            # Handle multi-level columns by flattening them
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = [
                    "_".join(str(col).strip() for col in cols if col)
                    for cols in pivot_df.columns.values
                ]

            # Add totals if requested
            if add_totals:
                # Add row totals
                numeric_cols = pivot_df.select_dtypes(
                    include=["number"]
                ).columns
                if len(numeric_cols) > 0:
                    pivot_df["Total"] = pivot_df[numeric_cols].sum(axis=1)

                # Add column totals
                totals_row = {}

                # Set index columns to "Total"
                for col in pivot_df.columns:
                    if col in index:
                        totals_row[col] = "Total"
                    elif col != "Total" and pd.api.types.is_numeric_dtype(
                        pivot_df[col]
                    ):
                        totals_row[col] = pivot_df[col].sum()
                    else:
                        totals_row[col] = None

                # If we added a row total column, calculate its total too
                if "Total" in pivot_df.columns:
                    totals_row["Total"] = pivot_df["Total"].sum()

                # Append the totals row
                pivot_df = pd.concat(
                    [pivot_df, pd.DataFrame([totals_row])], ignore_index=True
                )

            memory.put_dataframe(output_df_name, pivot_df, config=config)
            success_msg = f"Pivot table created successfully and stored as '{output_df_name}'"
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, pivot_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return PivotTable, short_description, long_description


def get_column_selection_tool(
    memory: Memory,
) -> Tuple[StructuredTool, str, str]:
    name = "ColumnSelection"
    short_description = "Tool to select a subset of columns from a dataframe."

    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    Name of the dataframe to query.
output_df_name : str
    Name of the new dataframe to create for the result
columns : List[str]
    The columns to select.

Returns
-------
A string indicating the result of column selection.
If failed, return the runtime exception.

Usage
-----
Call this tool to extract a subset of columns like ColumnSelection(
    dataframe_name='df',
    output_df_name='df_subset',
    columns=['col1', 'col2']
)
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def ColumnSelection(
        dataframe_name: Annotated[
            str, "Name of the dataframe in which to apply an operation."
        ],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result."
        ],
        columns: Annotated[
            List[str], "The arithmetic expression to evaluate as a string."
        ],
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            df = memory.get_dataframe(dataframe_name, config=config)
            if df is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            for col in columns:
                if col not in df.columns:
                    raise ValueError(
                        f"column {col} does NOT exist in dataframe {dataframe_name}"
                    )

            out_df = df[columns]
            memory.put_dataframe(output_df_name, out_df, config=config)
            success_msg = f"The new dataframe {output_df_name} has been created with columns {out_df.columns}."
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, out_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return ColumnSelection, short_description, long_description


def get_query_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "QueryDataframe"

    short_description = "Tool to query the columes of a DataFrame with a boolean expression using pandas.Dataframe.query(expression)"

    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    Name of the dataframe to query.
output_df_name : str
    Name of the new dataframe to create for the result
expression : str
    The boolean expression string to evaluate.

Returns
-------
A string indicating the result of dataframe querying.
If failed, return the runtime exception.

Usage
-----
Use this tool to filter rows with certain column condition.

Examples
--------
>>> df
    A   B  C C
0  1  10   10
1  2   8    9
2  3   6    8
3  4   4    7
4  5   2    6

Select rows where column A is larger than column B

>>> QueryDataframe(
...     dataframe_name='df',
...     output_df_name='df_A_larger_than_B',
...     expression='A > B'
... )
>>> print(df_A_larger_than_B)
    A  B  C C
4  5  2    6

For columns with spaces in their name, you can use backtick quoting. E.g, to select rows where B is equal to "C C"

>>> QueryDataframe(
...     dataframe_name='df',
...     output_df_name='df_B_equal_to_CC',
...     expression='B == `C C`'
... )
>>> print(df_B_equal_to_CC)
    A   B  C C
0  1  10   10
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def QueryDataframe(
        dataframe_name: Annotated[str, "Name of the dataframe to query."],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result."
        ],
        expression: Annotated[
            str, "The boolean expression to query a dataframe."
        ],
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            # Use pandas.eval to compute the result of the expression
            dataframe = memory.get_dataframe(dataframe_name, config=config)
            if dataframe is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            new_values = dataframe.query(expression, inplace=False)
            if not isinstance(new_values, pd.DataFrame):
                raise ValueError(
                    f"Failed to generate a new dataframe after querying with expression {expression}, the type of output is in the type of {type(new_values)}."
                )

            memory.put_dataframe(output_df_name, new_values, config=config)
            success_msg = f"After querying, a new dataframe created and stored as '{output_df_name}' with columns {', '.join(new_values.columns)}"
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, new_values)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return QueryDataframe, short_description, long_description


def get_concatenate_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "ConcatenateDataframes"
    short_description = (
        "Tool to concatenate two dataframes along columns or rows."
    )

    long_description = f"""
{short_description}

Parameters
----------
left_dataframe_name : str
    Name of the left dataframe to concatenate.
right_dataframe_name : str
    Name of the right dataframe to concatenate.
output_df_name : str
    Name of the new dataframe to create for the result
axis : int
    Axis along which to concatenate (0 for rows, 1 for columns). Default is 1.

Returns
-------
A string indicating the result of the concatenation operation.
If failed, return the runtime exception.

Usage
-----
- Use this tool to concatenate two dataframes along columns or rows. This tool is useful when you want to combine dataframes that do not have common columns to join on but can be aligned by their indices.
- Example: Concatenating two dataframes with different columns but the same number of rows.
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def ConcatenateDataframes(
        left_dataframe_name: Annotated[
            str, "Name of the left dataframe to concatenate"
        ],
        right_dataframe_name: Annotated[
            str, "Name of the right dataframe to concatenate"
        ],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result"
        ],
        axis: Annotated[
            int,
            "Axis along which to concatenate (0 for rows, 1 for columns). Default is 1.",
        ] = 1,
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            left_df = memory.get_dataframe(left_dataframe_name, config=config)
            right_df = memory.get_dataframe(right_dataframe_name, config=config)

            if left_df is None:
                raise no_dataframe_message(left_dataframe_name)
            if right_df is None:
                raise no_dataframe_message(right_dataframe_name)

            concatenated_df = pd.concat([left_df, right_df], axis=axis)

            memory.put_dataframe(output_df_name, concatenated_df, config=config)
            success_msg = f"Concatenated dataframe created and stored as '{output_df_name}'"
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, concatenated_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return ConcatenateDataframes, short_description, long_description


def get_merge_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "MergeDataframes"
    short_description = (
        "Tool to merge two dataframes based on a common column or index."
    )
    long_description = f"""
{short_description}

Parameters
----------
left_dataframe_name : str
    Name of the left dataframe to merge.
right_dataframe_name : str
    Name of the right dataframe to merge.
output_df_name : str
    Name of the new dataframe to create for the result.
how : str
    Type of merge to perform ('left', 'right', 'outer', 'inner'). Default is 'inner'.
left_on : Union[str, List[str]],
    One or a list of columns from the left dataframe to join on. Optional if using the index.
right_on : Union[str, List[str]],
    One or a list of columns from the right dataframe to join on. Optional if using the index.
left_index : bool
    Whether to use the index from the left dataframe as the join key. Default is False.
right_index : bool
    Whether to use the index from the right dataframe as the join key. Default is False.

Returns
-------
A string indicating the result of the merging operation.
If failed, return runtime exception.
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def MergeDataframes(
        left_dataframe_name: Annotated[
            str, "Name of the left dataframe to merge"
        ],
        right_dataframe_name: Annotated[
            str, "Name of the right dataframe to merge"
        ],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result"
        ],
        how: Annotated[
            str,
            "Type of merge to perform ('left', 'right', 'outer', 'inner'). Default is 'inner'",
        ] = "inner",
        left_on: Annotated[
            Union[str, List[str]],
            "One or a list of columns from the left dataframe to join on. Optional if using the index.",
        ] = None,
        right_on: Annotated[
            Union[str, List[str]],
            "One or a list of columns from the right dataframe to join on. Optional if using the index.",
        ] = None,
        left_index: Annotated[
            bool,
            "Whether to use the index from the left dataframe as the join key. Default is False.",
        ] = False,
        right_index: Annotated[
            bool,
            "Whether to use the index from the right dataframe as the join key. Default is False.",
        ] = False,
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            left_df = memory.get_dataframe(left_dataframe_name, config=config)
            right_df = memory.get_dataframe(right_dataframe_name, config=config)

            if left_df is None:
                raise ValueError(no_dataframe_message(left_dataframe_name))
            if right_df is None:
                raise ValueError(no_dataframe_message(right_dataframe_name))

            merged_df = pd.merge(
                left_df,
                right_df,
                how=how,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
            )

            memory.put_dataframe(output_df_name, merged_df, config=config)
            success_msg = (
                f"Merged dataframe created and stored as '{output_df_name}'"
            )
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, merged_df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return MergeDataframes, short_description, long_description


def get_column_calculator_tool(
    memory: Memory,
) -> Tuple[StructuredTool, str, str]:
    name = "CalculatorTool"
    short_description = (
        "Tool to evaluate arithmetic expressions using pandas.eval."
    )
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    Name of the dataframe in which to apply an operation.
output_df_name : str
    Name of the new dataframe to create for the result.
expression : str
    The arithmetic expression to evaluate as a string.

Returns
-------
A string indicating the result of the evaluation or an error message if the evaluation fails.
If failed, return the runtime exception.

Examples
--------
>>> df
    col1  col2
0     2     9
1     2     4
2     1     7
3     8     6
4     8    10
5     8    12

Calculate col3 = col1 * 2 + col2 and col4 as col4 = col3 * col3

>>> CalculatorTool(dataframe_name="df", output_dfd_name="df_calc", expression="col3 = col1 * 2 + col2\ncol4 = col3 * col3")
>>> print(df_calc)
    col1  col2  col3  col4
0     2     9    13   169
1     2     4     8    64
2     1     7     9    81
3     8     6    22   484
4     8    10    26   676
5     8    12    28   784

Note that, "expression" can have multiple lines. But each line should be a standalone expression with an output variable e.g. "X = ..."
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def ColumnCalculatorTool(
        dataframe_name: Annotated[
            str, "Name of the dataframe in which to apply an operation."
        ],
        output_df_name: Annotated[
            str, "Name of the new dataframe to create for the result."
        ],
        expression: Annotated[
            str, "The arithmetic expression to evaluate as a string."
        ],
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            # Use pandas.eval to compute the result of the expression
            dataframe = memory.get_dataframe(dataframe_name, config=config)
            if dataframe is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            new_values = dataframe.eval(expression)
            memory.put_dataframe(output_df_name, new_values, config=config)
            success_msg = f"After calculation, a new dataframe created and stored as '{output_df_name}' with columns {', '.join(new_values.columns)}"
            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, new_values)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return ColumnCalculatorTool, short_description, long_description


def get_absolute_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "AbsoluteTool"
    short_description = "Tool to compute the absolute value of a given input column and store the result in a new output column."
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    The name of the dataframe to be used.
input_column : str
    The column in the dataframe from which to compute the absolute values.
output_column : str
    The new column name where the absolute values will be stored.

Returns
-------
A string indicating the result of the operation or an error message if the operation fails.
If failed, return the runtime exception.

Examples
--------
AbsoluteTool('data', 'price', 'abs_price')
""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def AbsoluteTool(
        dataframe_name: Annotated[str, "The name of the dataframe to be used."],
        input_column: Annotated[
            str,
            "The column in the dataframe from which to compute the absolute values.",
        ],
        output_column: Annotated[
            str, "The new column name where the absolute values will be stored."
        ],
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            # Retrieve the dataframe from memory
            df = memory.get_dataframe(dataframe_name, config=config)
            if df is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            # Check if the input column exists in the dataframe
            if input_column not in df.columns:
                raise ValueError(
                    f"Column '{input_column}' not found in the dataframe."
                )

            # Compute the absolute value of the input column and assign it to the new output column
            df[output_column] = df[input_column].abs()

            # Store the updated dataframe back with the same name (or optionally with a new name if desired)
            memory.put_dataframe(dataframe_name, df, config=config)
            success_msg = f"Absolute values computed and stored in column '{output_column}' in dataframe '{dataframe_name}'."
            return f"{success_msg}\n{get_df_tool_message(memory, dataframe_name, df)}"
        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return AbsoluteTool, short_description, long_description


def get_unique_tool(memory: Memory) -> Tuple[StructuredTool, str, str]:
    name = "GetUniqueValue"
    short_description = "Return the unique values of the selected column."
    long_description = f"""
{short_description}

Parameters
----------
dataframe_name : str
    Dataframe to get unique values from.
output_df_name : str
    Name of the filtered unique values as a dataframe.
column : str
    The column to find unique values.

Returns
-------
A string indicating the result of the operation or an error message if the operation fails.
If failed, return the runtime exception.

Examples
--------
GetUniqueValue('df_data', 'col_one', 'df_unique')

""".strip()

    @tool(
        parse_docstring=False,
        description=long_description,
        infer_schema=True,
    )
    async def GetUniqueValue(
        dataframe_name: Annotated[str, "Dataframe to get unique values from."],
        output_df_name: Annotated[
            str, "Name of the filtered unique values as a dataframe."
        ],
        column: Annotated[str, "The column to find unique values."],
        config: Annotated[
            RunnableConfig, "Langchain Runnable Configuration"
        ] = {},
    ) -> str:
        try:
            df = memory.get_dataframe(dataframe_name, config=config)
            if df is None:
                raise ValueError(no_dataframe_message(dataframe_name))

            if column not in df.columns:
                raise ValueError(
                    f"Column {column} does NOT exist in dataframe {dataframe_name}."
                )

            out_col = f"Unique {column}"
            out_df = pd.DataFrame({out_col: df[column].unique()})

            memory.put_dataframe(output_df_name, out_df, config)

            success_msg = f"The unique values of column {column} from dataframe {output_df_name} have been generated, and saved as column '{out_col}' in a new dataframe {output_df_name}."

            return f"{success_msg}\n{get_df_tool_message(memory, output_df_name, out_df)}"

        except Exception as e:
            return f"Error in calling tool {name} with the following exception: {repr(e)}"

    return GetUniqueValue, short_description, long_description


DEFAULT_ANALYTICS_TOOLS = {
    "ColumnCalculatorTool": get_column_calculator_tool,
    "QueryDataframe": get_query_tool,
    "ColumnSelection": get_column_selection_tool,
    "GroupBy": get_groupby_tool,
    "ColumnAggregation": get_aggregrate_tool,
    "MergeDataframes": get_merge_tool,
    "ConcatenateDataframes": get_concatenate_tool,
    "PivotTable": get_pivot_tool,
    "SortValue": get_sort_value_tool,
    "nLargest": get_n_largest_tool,
    "nSmallest": get_n_smallest_tool,
    "GetUniqueValue": get_unique_tool,
    "AbsoluteTool": get_absolute_tool,
    "CalculateCorrelationMatrix": get_correlation_matrix_tool,
}