from typing import List, Literal, Union, Dict, Tuple
from langchain.tools import tool, StructuredTool # Assuming StructuredTool is the intended return type for get_..._tool
from dataqalib.memory import Memory # Assuming this is the correct Memory class

# Tool: CalculateCorrelationMatrix
def get_correlation_matrix_tool(memory: Memory) -> StructuredTool:
    name = "CalculateCorrelationMatrix"

    @tool
    def CalculateCorrelationMatrix(
        df_name: str,
        output_df_name: str,
        method: Literal['pearson', 'kendall', 'spearman'] = 'pearson',
        min_periods: int = 1,
        numeric_only: bool = False
    ) -> str:
        """
        Compute pairwise correlation of columns for dataframe called `df_name`, excluding NA/null values, save the correlation matrix as a new dataframe called `output_df_name`.

        Parameters
        ----------
        df_name : str
            The name of the dataframe to calculate correlation.
        output_df_name : str
            The name of the correlation matrix as a dataframe.
        method : {'pearson', 'kendall', 'spearman'} or callable, default 'pearson'
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
        Tool calling response: str
            - If successful, return a message saying that "The correlation matrix of dataframe `df_name` has been calculated and saved in a new dataframe `output_df_name`."
            - If failed, return a message of the runtime exception.

        Example
        -------
        Assume that we have a dataframe called "df_abc" with 5 rows and 3 columns A, B, C.
        >>> print(df_abc)
             A         B         C
        0  0.655982  0.998371  0.431369
        1  0.093596  0.595080  0.873763
        2  0.379816  0.065121  0.772393
        3  0.479515  0.020517  0.0855805
        4  0.433931  0.045164  0.734673

        Calculate the correlation matrix of df_abc in a dataframe df_abc_corr
        >>> CalculateCorrelationMatrix(df_name="df_abc", output_df_name="df_abc_corr")
        >>> print(df_abc_corr)
                  A         B         C
        A  1.000000  0.614068 -0.613955
        B  0.614068  1.000000 -0.208519
        C -0.613955 -0.208519  1.000000
        """
        try:
            df = memory.get_dataframe(df_name)
            out_df = df.corr(method=method, min_periods=min_periods, numeric_only=numeric_only)
            memory.put_dataframe(output_df_name, out_df)
            return f"DataFrame {output_df_name} has been successfully generated as the correlation matrix of {df_name}."
        except Exception as e:
            return f"Tool {name} failed with the following exception\n{str(e)}"
    # The @tool decorator makes CalculateCorrelationMatrix callable and assigns attributes.
    # If explicit naming is needed for registration or consistency:
    # CalculateCorrelationMatrix.name = name 
    return CalculateCorrelationMatrix

# Tool: nLargest
def get_n_largest_tool(memory: Memory) -> StructuredTool:
    name = "nLargest"

    @tool
    def nLargest(
        df_name: str,
        output_df_name: str,
        n: int,
        columns: Union[str, List[str]],
        keep: Literal['first', 'last', 'all'] = "first"
    ) -> str:
        """
        Return the first `n` rows with the largest values in `columns`, in descending order. The columns that are not specified are returned as well, but not used for ordering.

        Parameters
        ----------
        df_name : str
            The name of the dataframe to get n-largest rows.
        output_df_name : str
            The name of n-largest rows as a dataframe.
        n : int
            Number of rows to return.
        columns : column name or list of column names
            Column label(s) to order by.
        keep : {'first', 'last', 'all'}, default 'first'
            Where there are duplicate values:
            - `first` : prioritize the first occurrence(s)
            - `last`  : prioritize the last occurrence(s)
            - `all`   : keep all the ties of the smallest item even if it means selecting more than ``n`` items.

        Returns
        -------
        Tool calling response: str
            - If successful, return a message saying that "N-largest rows of dataframe `df_name` has been calculated and saved in a new dataframe `output_df_name`."
            - If failed, return a message of the runtime exception.

        Example
        -------
        Assume that we have a dataframe called "df_country"
        >>> print(df_country)
                   population      GDP alpha-2
        Italy      59000000  1937894        IT
        Malta        434000    12011        MT
        Maldives     434000     4520        MV
        Iceland      337000    17036        IS

        Select two countries with the largest population
        >>> nLargest(df_name="df_country", output_df_name="df_top_2_population", n=2, columns="population")
        >>> print(df_top_2_population)
                   population      GDP alpha-2
        Italy      59000000  1937894        IT
        Malta        434000    12011        MT

        When using ``keep='last'``, ties are resolved in reverse order:
        >>> nLargest(df_name="df_country", output_df_name="df_top_2_population", n=2, columns="population", keep="last")
        >>> print(df_top_2_population)
                   population      GDP alpha-2
        Italy      59000000  1937894        IT
        Malta        434000    12011        MT

        When using ``keep='all'``, the number of element kept can go beyond ``n``
        if there are duplicate values for the largest element, all the
        ties are kept:
        >>> nLargest(df_name="df_country", output_df_name="df_top_2_population", n=2, columns="population", keep="all")
        >>> print(df_top_2_population)
                   population      GDP alpha-2
        Italy      59000000  1937894        IT
        Malta        434000    12011        MT
        Maldives     434000     4520        MV
        """
        try:
            df = memory.get_dataframe(df_name)
            out_df = df.nlargest(n, columns=columns, keep=keep)
            memory.put_dataframe(output_df_name, out_df)
            return f"Top {n} largest rows of {df_name} has been calculated and saved in a new dataframe {output_df_name}."
        except Exception as e:
            return f"Tool {name} failed with the following exception\n{str(e)}"
    return nLargest

# Tool: nSmallest
def get_n_smallest_tool(memory: Memory) -> StructuredTool:
    name = "nSmallest"

    @tool
    def nSmallest(
        df_name: str,
        output_df_name: str,
        n: int,
        columns: Union[str, List[str]],
        keep: Literal['first', 'last', 'all'] = "first"
    ) -> str:
        """
        Return the first `n` rows with the smallest values in `columns`, in ascending order. The columns that are not specified are returned as well, but not used for ordering.

        Parameters
        ----------
        df_name : str
            The name of the dataframe to get n-smallest rows.
        output_df_name : str
            The name of n-smallest rows as a dataframe.
        n : int
            Number of rows to return.
        columns : column name or list of column names
            Column label(s) to order by.
        keep : {'first', 'last', 'all'}, default 'first'
            Where there are duplicate values:
            - `first` : prioritize the first occurrence(s)
            - `last`  : prioritize the last occurrence(s)
            - `all`   : keep all the ties of the smallest item even if it means selecting more than ``n`` items.

        Returns
        -------
        Tool calling response: str
            - If successful, return a message saying that "N-smallest rows of dataframe `df_name` has been calculated and saved in a new dataframe `output_df_name`."
            - If failed, return a message of the runtime exception.

        Example
        -------
        Assume that we have a dataframe called "df_country"
        >>> print(df_country)
                   population      GDP alpha-2
        Italy      59000000  1937894        IT
        Malta        434000    12011        MT
        Maldives     434000     4520        MV
        Iceland      337000    17036        IS

        Select two countries with the smallest population
        >>> nSmallest(df_name="df_country", output_df_name="df_top_2_population", n=2, columns="population")
        >>> print(df_top_2_population)
                   population      GDP alpha-2
        Iceland      337000    17036        IS
        Malta        434000    12011        MT

        When using ``keep='last'``, ties are resolved in reverse order:
        >>> nSmallest(df_name="df_country", output_df_name="df_top_2_population", n=2, columns="population", keep="last")
        >>> print(df_top_2_population)
                   population      GDP alpha-2
        Iceland      337000    17036        IS
        Maldives     434000     4520        MV

        When using ``keep='all'``, the number of element kept can go beyond ``n``
        if there are duplicate values for the largest element, all the
        ties are kept:
        >>> nSmallest(df_name="df_country", output_df_name="df_top_2_population", n=2, columns="population", keep="all")
        >>> print(df_top_2_population)
                   population      GDP alpha-2
        Iceland      337000    17036        IS
        Malta        434000    12011        MT
        Maldives     434000     4520        MV
        """
        try:
            df = memory.get_dataframe(df_name)
            out_df = df.nsmallest(n, columns=columns, keep=keep)
            memory.put_dataframe(output_df_name, out_df)
            return f"Top {n} smallest rows of {df_name} has been calculated and saved in a new dataframe {output_df_name}."
        except Exception as e:
            return f"Tool {name} failed with the following exception\n{str(e)}"
    return nSmallest

# Tool: SortValue
def get_sort_value_tool(memory: Memory) -> StructuredTool:
    name = "SortValue"

    @tool
    def SortValue(
        df_name: str,
        output_df_name: str,
        by: Union[str, List[str]],
        axis: Union[int, Literal['index', 'columns']] = 0,
        ascending: Union[bool, List[bool], Tuple[bool, ...]] = True,
    ) -> str:
        """
        Sort by the values along either axis.

        Parameters
        ----------
        df_name : str
            The name of the dataframe to get n-smallest rows.
        output_df_name : str
            The name of n-smallest rows as a dataframe.
        by : str or list of str
            Name or list of names to sort by.
            - if `axis` is 0 or 'index' then `by` may contain index levels and/or column labels.
            - if `axis` is 1 or 'columns' then `by` may contain column levels and/or index labels.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis to be sorted.
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort orders. If this is a list of bools, must match the length of the by.

        Returns
        -------
        Tool calling response: str
            - If successful, return a message saying that "The sorted dataframe `df_name` has been created and saved as a new dataframe `output_df_name`."
            - If failed, return a message of the runtime exception.

        Example
        -------
        >>> df
          col1  col2 col3 col4
        0    A     2    0    a
        1    A     1    1    b
        2    B     9    9    c
        3  NaN     8    4    d
        4    D     7    2    e
        5    C     4    3    f

        Sort by col1
        >>> SortValue(df_name='df', output_df_name='df_sort', by=['col1'])
        >>> print(df_sort)
          col1  col2 col3 col4
        0    A     2    0    a
        1    A     1    1    b
        2    B     9    9    c
        5    C     4    3    f
        4    D     7    2    e
        3  NaN     8    4    d
        """
        try:
            df = memory.get_dataframe(df_name)
            out_df = df.sort_values(by=by, axis=axis, ascending=ascending)
            memory.put_dataframe(output_df_name, out_df)
            return f"The sorted dataframe {df_name} has been created and saved as a new dataframe {output_df_name}."
        except Exception as e:
            return f"Tool {name} failed with the following exception\n{str(e)}"
    return SortValue

DEFAULT_ANALYTICS_TOOLS = {
    'CalculateCorrelationMatrix': get_correlation_matrix_tool,
    'nLargest': get_n_largest_tool,
    'nSmallest': get_n_smallest_tool,
    'SortValue': get_sort_value_tool,
}

def get_analytics_tools(
    memory: Memory,
    tool_names: Union[List[str], Dict[str, callable]] = DEFAULT_ANALYTICS_TOOLS, # Made Dict more specific
) -> List[StructuredTool]: # Assuming the goal is a list of tool instances
    tools = []
    # If tool_names is a list, iterate through it. If it's a dict (like default), iterate its keys.
    names_to_load = tool_names if isinstance(tool_names, list) else tool_names.keys()
    
    for name in names_to_load:
        if name not in DEFAULT_ANALYTICS_TOOLS:
            raise ValueError(f"Tool {name} is not defined.")
        tool_factory = DEFAULT_ANALYTICS_TOOLS[name]
        tools.append(tool_factory(memory=memory))
    return tools