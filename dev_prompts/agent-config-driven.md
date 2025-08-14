Okay, let's define a configuration structure for the CwdAgent. This configuration will be user-centric, allowing users to define key aspects of the agent's behavior without needing to delve into the code. We'll provide Pydantic models for this configuration and an example YAML.

1. Pydantic Models for Configuration

We'll create a new file, for example, dataqa/agent/cwd_agent/config.py, to house these Pydantic models.
```python
# dataqa/agent/cwd_agent/config.py
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any

# Re-using prompt_type definition style for flexibility in prompt content
# Or defining a more structured way for message lists.
class PromptMessageConfig(BaseModel):
    role: str = Field(default="system", description="Role of the message (e.g., 'system', 'user', 'assistant')")
    content: str = Field(description="Content of the message. Can use {placeholders} and <schema>.")

CwdAgentPromptValue = Union[str, List[PromptMessageConfig]] # A single string (treated as system message content) or a list of messages

class CwdAgentPromptsConfig(BaseModel):
    planner_prompt: CwdAgentPromptValue
    replanner_prompt: CwdAgentPromptValue
    sql_generator_prompt: CwdAgentPromptValue = Field(description="Prompt for the RetrievalWorker's SQLGenerator.")
    analytics_prompt: CwdAgentPromptValue = Field(description="Prompt for the AnalyticsWorker.")
    plot_prompt: CwdAgentPromptValue = Field(description="Prompt for the PlotWorker.")

class UserSqlExecutionDataFileConfig(BaseModel):
    path: str = Field(description="Path to the data file (e.g., CSV, Excel).")
    table_name: str = Field(description="Name to assign to the table in the in-memory SQL database.")

class UserInMemorySqlExecutorConfig(BaseModel):
    data_files: List[UserSqlExecutionDataFileConfig] = Field(
        description="List of data files to load into the in-memory SQL database."
    )

class RetrievalWorkerAgentConfig(BaseModel):
    sql_execution_config: UserInMemorySqlExecutorConfig

class AnalyticsWorkerAgentConfig(BaseModel):
    # Future: enabled_tools: Optional[List[str]] = None
    pass # Currently uses default tools; prompt is main config

class PlotWorkerAgentConfig(BaseModel):
    # Future: configuration for plotting tools if needed
    pass # Prompt is main config

class CwdAgentWorkersModulesConfig(BaseModel):
    retrieval_worker: RetrievalWorkerAgentConfig
    analytics_worker: Optional[AnalyticsWorkerAgentConfig] = Field(default_factory=AnalyticsWorkerAgentConfig)
    plot_worker: Optional[PlotWorkerAgentConfig] = Field(default_factory=PlotWorkerAgentConfig)

class LLMSelectionConfig(BaseModel):
    type: str = Field(description="Fully qualified class name for the LLM (e.g., 'dataqa.llm.openai.AzureOpenAI').")
    config: Dict[str, Any] = Field(description="Configuration dictionary for the chosen LLM type (e.g., model, api_key, base_url).")

class CwdAgentDefinitionConfig(BaseModel):
    agent_name: Optional[str] = Field(default="CwdAgent", description="An optional name for this agent configuration.")
    llm: LLMSelectionConfig = Field(description="LLM configuration to be used by the agent and its sub-components.")
    schema_info: str = Field(
        description="A string containing database schema information. This will replace the '<schema>' placeholder in prompts."
    )
    prompts: CwdAgentPromptsConfig = Field(description="Definitions for all prompts used by the agent's components.")
    workers: CwdAgentWorkersModulesConfig = Field(description="Configuration for the agent's specialized workers.")

    class Config:
        extra = 'forbid' # To catch typos in config file keys
```
2. Example YAML Configuration File

A user would create a YAML file (e.g., my_cwd_agent_config.yaml) based on the structure defined by CwdAgentDefinitionConfig.

# my_cwd_agent_config.yaml
```yaml
agent_name: "SalesDataQAAgent"

llm:
  type: "dataqa.llm.openai.AzureOpenAI"
  config:
    model: "gpt-4o-mini" # Example model
    api_version: "2023-07-01-preview"
    # It's highly recommended to set api_key and base_url via environment variables
    # or a secure secrets management solution, not directly in this file.
    # If AzureOpenAIConfig model supports reading from env vars, that's ideal.
    # api_key: ${AZURE_OPENAI_API_KEY} # Example if Pydantic model supports env var substitution
    # base_url: ${AZURE_OPENAI_ENDPOINT}
    temperature: 0.5
    max_response_token: 1500

schema_info: |
  Table: sales
  Columns:
    - order_id (INTEGER, PRIMARY KEY)
    - product_name (VARCHAR)
    - quantity (INTEGER)
    - unit_price (DECIMAL)
    - order_date (DATE)
    - region (VARCHAR)

  Table: products
  Columns:
    - product_id (INTEGER, PRIMARY KEY)
    - product_name (VARCHAR, UNIQUE)
    - category (VARCHAR)

prompts:
  planner_prompt:
    # Can be a single string (system message content)
    # "You are a master planner..."
    # Or a list of messages:
    - role: "system"
      content: |
        You are a planner. Your goal is to create a step-by-step plan to answer the user's query: {query}.
        You have access to the following dataframes (initially none, will be populated by RetrievalWorker):
        {dataframe_summary}
        Available workers and their functions:
        - RetrievalWorker: Executes SQL queries to fetch data into dataframes. Use for questions requiring data from the database.
        - AnalyticsWorker: Performs pandas-based operations on existing dataframes (correlation, n-largest, sorting, etc.).
        - PlotWorker: Generates plots from dataframes.

        Database Schema:
        <schema>

        Analyze the query and create a concise plan.
        Output a plan with tasks, specifying the worker and a clear task_description for each step.
        Example Task:
        { "worker": "RetrievalWorker", "task_description": "Retrieve total sales for each product" }

  replanner_prompt:
    - role: "system"
      content: |
        You are a replanner. Your role is to evaluate the progress and adjust the plan if necessary.
        Original user query: {query}
        Current plan state:
        {plan}
        Summary of completed steps and their outputs:
        {past_steps}
        Currently available dataframes and their summaries:
        {dataframe_summary}
        Database Schema:
        <schema>

        Review the original query, the plan, and the results of past steps.
        - If the current plan is sufficient and the next step is clear, keep the plan.
        - If a step failed or new information requires a change, revise the plan.
        - If all necessary information to answer the query has been gathered and processed, formulate a final response to the user using the Response tool.
        - If more steps are needed, update the plan with new tasks.
        Output either a new Plan or a final Response.

  sql_generator_prompt:
    - role: "system"
      content: |
        You are an expert SQL generator. Given the user's request and the database schema, generate an efficient SQL query.
        User request (task): {query} # This will be the task_description for RetrievalWorker
        Database Schema:
        <schema>

        Instructions:
        1. Understand the request and identify the required tables and columns.
        2. Construct a syntactically correct SQL query for DuckDB.
        3. Provide a descriptive name for the output dataframe where the query results will be stored. This name should be a valid Python variable name (e.g., `df_sales_summary`).

        Output format must be a JSON object with "sql", "reasoning", and "output" keys.
        Example:
        {
          "sql": "SELECT product_name, SUM(quantity * unit_price) AS total_revenue FROM sales GROUP BY product_name ORDER BY total_revenue DESC LIMIT 10;",
          "reasoning": "The user wants the top 10 products by revenue. I need to join sales data, calculate revenue per product, group by product, order by revenue, and limit to 10. The output dataframe will be named 'df_top_products'.",
          "output": "df_top_products"
        }

  analytics_prompt:
    - role: "system"
      content: |
        You are an analytics expert. You have access to pandas DataFrames in memory and a suite of tools to manipulate them.
        Current high-level task: {task}
        Overall plan:
        {plan}
        Available dataframes:
        {dataframe_summary}

        Your goal is to execute the current analytics task using the available tools.
        Think step-by-step. Clearly state your reasoning and the tool calls you make.
        The tools operate on dataframes in memory. Make sure to specify correct input and output dataframe names.

  plot_prompt:
    - role: "system"
      content: |
        You are a data visualization expert. Your task is to generate Python code for plotting based on the user's request.
        Current plotting task: {task}
        Overall plan:
        {plan}
        Available dataframes for plotting:
        {dataframe_summary}

        Instructions:
        1. Use Matplotlib or Seaborn for plotting.
        2. Generate complete, executable Python code.
        3. Assume necessary libraries (pandas, matplotlib.pyplot, seaborn) are imported.
        4. The code should save the plot to a file (e.g., 'plot.png') or produce bytes that can be displayed.
        5. Ensure your code is well-commented.

        Example for a bar chart of 'df_example' with columns 'category' and 'value':
        ```python
        import matplotlib.pyplot as plt
        import pandas as pd

        # Assuming df_example is a pandas DataFrame available in the execution environment
        # df_example = memory.get_dataframe('df_example') # This line is illustrative of how it might be accessed

        plt.figure(figsize=(10, 6))
        plt.bar(df_example['category'], df_example['value'])
        plt.xlabel('Category')
        plt.ylabel('Value')
        plt.title('Values per Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # To save the plot:
        # plt.savefig('my_plot.png')
        # To return bytes (more complex, usually handled by a tool wrapper):
        # import io
        # img_bytes = io.BytesIO()
        # plt.savefig(img_bytes, format='png')
        # img_bytes.seek(0)
        # plot_output = img_bytes.read() # This would be the tool's return
        plt.show() # For interactive, but for agents, savefig or byte stream is better.
        ```
        Focus on generating the plotting commands. The execution environment will handle dataframe access.

workers:
  retrieval_worker:
    sql_execution_config:
      data_files:
        - path: "data/sales_data.csv" # Relative or absolute path to your data
          table_name: "sales"
        - path: "data/product_catalog.xlsx"
          table_name: "products"

  analytics_worker: {} # No specific config other than prompt for now

  plot_worker: {} # No specific config other than prompt for now
```

3. Modifying CwdAgent to Use the Configuration

The CwdAgent class would need to be adapted to accept this configuration object instead of individual parameters like llm, prompts, and sql_execution_config.

Conceptual changes in dataqa/agent/cwd_agent/cwd_agent.py:
```python
from dataqa.memory import Memory
from dataqa.agent.base import Agent
from dataqa.llm.base_llm import BaseLLM
# New import for the configuration model
from .config import CwdAgentDefinitionConfig, CwdAgentPromptsConfig, PromptMessageConfig, CwdAgentPromptValue
from dataqa.components.code_executor.in_memory_code_executor import InMemoryCodeExecutorConfig
from dataqa.components.base_component import Variable # For defining fixed inputs to InMemoryCodeExecutorConfig
from dataqa.utils.utils import cls_from_str # For dynamic class instantiation
from dataqa.utils.prompt_utils import build_prompt, prompt_type # For building prompts
# ... other existing imports ...

class CwdAgent(Agent):
    def __init__(self,
                 memory: Memory, # Memory is still passed in directly
                 config: CwdAgentDefinitionConfig # Agent configuration object
                 ):
        self.agent_config = config # Store the raw config if needed

        # 1. Instantiate LLM from config
        llm_cls = cls_from_str(config.llm.type)
        # The specific LLM class (e.g., AzureOpenAI) has its own config model (e.g., AzureOpenAIConfig)
        llm_instance_config_model = llm_cls.config_base_model
        llm_specific_config_obj = llm_instance_config_model(**config.llm.config)
        llm = llm_cls(config=llm_specific_config_obj)

        # 2. Preprocess prompts with schema_info
        # This method will take CwdAgentPromptsConfig and schema_info,
        # and return a Dict[str, prompt_type] suitable for Planner, Workers, etc.
        self.processed_prompts = self._process_prompts_from_config(
            prompts_config=config.prompts,
            schema_info=config.schema_info
        )

        # 3. Prepare InMemoryCodeExecutorConfig for RetrievalWorker
        # The user provides data_files. We construct the full InMemoryCodeExecutorConfig.
        # For the SQLExecutor within RetrievalWorker, input/output schemas are internally defined.
        self.retrieval_sql_exec_config = InMemoryCodeExecutorConfig(
            name=f"{config.agent_name}_RetrievalSqlExecutor" if config.agent_name else "CwdAgent_RetrievalSqlExecutor",
            data_files=[df.model_dump() for df in config.workers.retrieval_worker.sql_execution_config.data_files],
            input=[], # SQLExecutor in RetrievalWorker uses a specific state, not generic component input mapping
            output=[] # SQLExecutor returns a specific dict, base CodeExecutorOutput is sufficient
        )

        # Initialize the base Agent class
        super().__init__(memory=memory, llm=llm) # llm is now instantiated from config

    def _process_prompts_from_config(self, prompts_config: CwdAgentPromptsConfig, schema_info: str) -> Dict[str, prompt_type]:
        """
        Processes prompts from the configuration, injects schema_info,
        and prepares them in a format (prompt_type) usable by build_prompt.
        """
        final_prompts: Dict[str, prompt_type] = {}
        for field_name, prompt_value_config in prompts_config: # Iterate through Pydantic model fields
            # prompt_value_config is CwdAgentPromptValue (Union[str, List[PromptMessageConfig]])
            
            processed_content: prompt_type
            if isinstance(prompt_value_config, str):
                # Single string treated as system message content
                processed_content = prompt_value_config.replace('<schema>', schema_info)
            elif isinstance(prompt_value_config, list):
                # List of PromptMessageConfig objects
                msg_list_for_build_prompt = []
                for msg_conf in prompt_value_config:
                    msg_dict = msg_conf.model_dump()
                    msg_dict['content'] = msg_dict['content'].replace('<schema>', schema_info)
                    msg_list_for_build_prompt.append(msg_dict)
                processed_content = msg_list_for_build_prompt
            else:
                # Should not happen if Pydantic validation is correct
                raise ValueError(f"Unexpected prompt configuration type for {field_name}")
            
            final_prompts[field_name] = processed_content
        return final_prompts

    def build_workflow(self, memory: Memory, llm: BaseLLM) -> CompiledGraph:
        self.planner = Planner(
            memory=memory, llm=llm,
            prompt=self.processed_prompts['planner_prompt']
        )
        self.replanner = Replanner(
            memory=memory, llm=llm,
            prompt=self.processed_prompts['replanner_prompt']
        )
        self.retrieval_worker = RetrievalWorker(
            memory=memory, llm=llm,
            sql_prompt=self.processed_prompts['sql_generator_prompt'],
            sql_execution_config=self.retrieval_sql_exec_config # Use the prepared config
        )
        self.analytics_worker = AnalyticsWorker(
            memory=memory, llm=llm,
            prompt=self.processed_prompts['analytics_prompt']
            # Pass config.workers.analytics_worker if it has more fields in the future
        )
        self.plot_worker = PlotWorker(
            memory=memory, llm=llm,
            prompt=self.processed_prompts['plot_prompt']
            # Pass config.workers.plot_worker if it has more fields
        )

        # ... rest of the graph definition (StateGraph, nodes, edges) remains the same ...
        workflow = StateGraph(CwdState)
        workflow.add_node("planner", self.planner)
        workflow.add_node("replanner", self.replanner) 
        workflow.add_node(WorkerName.RetrievalWorker.value, self.retrieval_worker)
        workflow.add_node(WorkerName.AnalyticsWorker.value, self.analytics_worker)
        workflow.add_node(WorkerName.PlotWorker.value, self.plot_worker)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "replanner")
        workflow.add_edge(WorkerName.RetrievalWorker.value, 'replanner')
        workflow.add_edge(WorkerName.AnalyticsWorker.value, 'replanner')
        workflow.add_edge(WorkerName.PlotWorker.value, 'replanner')
        workflow.add_conditional_edges("replanner", task_router) # Note: Original had planner and replanner for conditional edges
                                                                 # This assumes replanner always routes after planner's first plan
                                                                 # and after each worker.

        return workflow.compile()

    @classmethod
    def from_config_path(cls, config_file_path: str, memory: Memory) -> "CwdAgent":
        """
        Factory method to create a CwdAgent instance from a YAML configuration file.
        """
        import yaml
        with open(config_file_path, 'r') as f:
            raw_config_dict = yaml.safe_load(f)
        
        agent_definition_config = CwdAgentDefinitionConfig(**raw_config_dict)
        return cls(memory=memory, config=agent_definition_config)

    # __call__ and display_workflow methods would remain largely the same.
    # The original _preprocess_prompts is replaced by _process_prompts_from_config.
```