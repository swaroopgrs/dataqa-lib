import asyncio
import yaml
import logging
import os
from pathlib import Path
from pprint import pprint

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dataqa.pipelines.config import PipelineConfig
from dataqa.pipelines.builder import build_graph_from_config
from dataqa.pipelines.state import PipelineInput
from dataqa.llm_providers.base import BaseLLMProvider # To ensure correct type loading
from dataqa.components.code_execution.in_memory import InMemoryCodeExecutor # Ensure components are importable
from dataqa.components.llm_query.prompt_chain import BasePromptLLMChain
from dataqa.components.prompt_templating.base import BasePromptComponent
from dataqa.components.flow_control.output_collector import OutputCollector



# Langchain imports
from langchain_core.runnables.config import RunnableConfig


async def run_pipeline(config_path: Path, base_dir: Path, initial_input: PipelineInput):
    """Loads config, builds graph, and runs the pipeline for a given input."""
    logger.info(f"--- Running pipeline for query: '{initial_input.query}' ---")

    # 1. Load YAML configuration
    logger.info(f"Loading configuration from: {config_path}")
    try:
        raw_config_content = config_path.read_text()
        # Resolve BASE_DIR placeholder
        resolved_config_content = raw_config_content.replace("{BASE_DIR}", str(base_dir))
        config_dict = yaml.safe_load(resolved_config_content)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading or parsing YAML configuration: {e}")
        return None

    # Check if Azure credentials need to be injected from environment if not in config
    # (This logic might be better placed within the builder/provider init)
    provider_def = next((item for item in config_dict.get('components', []) if item["name"] == "gpt_4o_model_provider"), None)
    if provider_def:
         if 'azure_endpoint' not in provider_def['params'] or not provider_def['params']['azure_endpoint']:
              provider_def['params']['azure_endpoint'] = os.getenv("AZURE_OPENAI_ENDPOINT")
              logger.info("Attempting to use AZURE_OPENAI_ENDPOINT environment variable.")
         if 'api_key' not in provider_def['params'] or not provider_def['params']['api_key']:
              provider_def['params']['api_key'] = os.getenv("AZURE_OPENAI_API_KEY")
              logger.info("Attempting to use AZURE_OPENAI_API_KEY environment variable.")
         # Basic check if credentials are still missing
         if not provider_def['params'].get('azure_endpoint') or not provider_def['params'].get('api_key'):
              logger.error("Azure OpenAI endpoint or API key is missing. Set in config or environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY).")
              return None


    # 2. Parse configuration
    logger.info("Parsing pipeline configuration...")
    try:
        pipeline_config_obj = PipelineConfig(**config_dict)
    except Exception as e: # Catch Pydantic ValidationError and others
        logger.error(f"Configuration validation failed: {e}")
        return None

    # 3. Build the graph
    logger.info("Building pipeline graph...")
    try:
        # Pass base_dir for resolving FILE_{BASE_DIR} paths
        compiled_graph, StateModel = build_graph_from_config(
            pipeline_config=pipeline_config_obj,
            pipeline_name="payments_pipeline", # Specify the pipeline to build
            base_dir=str(base_dir)
            # checkpointer=MemorySaver() # Optional: Add if state tracking needed
        )
        logger.info("Graph built successfully.")
    except Exception as e:
        logger.exception(f"Failed to build pipeline graph: {e}")
        return None

    # 4. Prepare initial state
    # The state model class 'StateModel' is returned by the builder
    # We only need to provide the 'input' field required by BasePipelineState
    initial_state = {"input": initial_input}
    logger.info(f"Initial pipeline input prepared: {initial_input.model_dump_json(indent=2)}")

    # 5. Invoke the graph
    logger.info("Invoking pipeline graph...")
    # Configure recursion limit for LangGraph if needed
    runtime_config = RunnableConfig(recursion_limit=15) # Adjust as needed
    try:
        final_state_dict = await compiled_graph.ainvoke(initial_state, config=runtime_config)
        logger.info("Pipeline execution completed.")
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        # Optionally try to extract partial state if available
        # final_state_dict = getattr(e, 'partial_state', initial_state) # Example, actual structure may vary
        return None # Indicate failure

    # 6. Process and display results
    logger.info("--- Pipeline Final State ---")
    # Validate final state for debugging
    try:
        final_state = StateModel.model_validate(final_state_dict)
        # Pretty print relevant parts of the final state
        print("\n--- Final Output ---")
        if final_state.final_output:
             pprint(final_state.final_output.model_dump(exclude_none=True), indent=2)
        else:
             print("Final output field is None.")

        if final_state.error:
             print("\n--- Pipeline Error ---")
             pprint(final_state.error.model_dump(exclude_none=True), indent=2)

        # Optionally print other intermediate states for debugging
        # print("\n--- Query Rewriter Output ---")
        # if final_state.query_rewriter_output:
        #     pprint(final_state.query_rewriter_output.model_dump(exclude_none=True))
        # print("\n--- Code Generator Output ---")
        # if final_state.code_generator_output:
        #      pprint(final_state.code_generator_output.model_dump(exclude_none=True))
        # print("\n--- Code Executor Output ---")
        # if final_state.code_executor_output:
        #      pprint(final_state.code_executor_output.model_dump(exclude_none=True))


    except Exception as e:
         logger.error(f"Failed to validate or display final state: {e}")
         print("\n--- Raw Final State Dictionary ---")
         pprint(final_state_dict, indent=2) # Print raw dict if validation fails

    return final_state_dict


async def main():
    # Define script path and base directory
    script_dir = Path(__file__).parent
    # Assumes config is in ./config and data is in ./data relative to the script
    # Assumes {BASE_DIR} in config refers to the parent of 'examples' directory
    config_file = script_dir / "config" / "config.yaml"
    base_directory = script_dir.parent.parent # Adjust if project structure is different

    # --- Example Questions ---
    # Example 1: Initial Question
    input1 = PipelineInput(
        query="What are the total sales for Starbucks?"
        # history=[], # Assuming empty history for first turn
        # previous_rewritten_query=None # Explicitly None for first turn
    )
    await run_pipeline(config_file, base_directory, input1)

    print("\n" + "="*50 + "\n")

    # Example 2: Follow-up Question (hypothetical - needs state management or manual context)
    # Note: Running a follow-up requires either a checkpointer in the graph build
    # or manually providing the 'previous_rewritten_query' from the first run's output.
    # Let's simulate providing context manually for this example.
    input2 = PipelineInput(
        query="Break it down by country.",
        # history=[...], # Add history if needed
        # Provide the rewritten query from the previous successful run
        previous_rewritten_query="What are the total sales for Starbucks?"
    )
    await run_pipeline(config_file, base_directory, input2)

    print("\n" + "="*50 + "\n")

    # Example 3: Different Company
    input3 = PipelineInput(
        query="Show me active Home Depot transaction divisions in California."
    )
    await run_pipeline(config_file, base_directory, input3)


if __name__ == "__main__":
    # Check if config file exists
    script_dir = Path(__file__).parent
    config_file_path = script_dir / "config" / "config.yaml"
    if not config_file_path.is_file():
        logger.error(f"Configuration file not found: {config_file_path}")
        logger.error("Please ensure 'config/config.yaml' exists relative to this script.")
    else:
        # Check if data files exist (relative to resolved BASE_DIR)
        base_dir_path = script_dir.parent.parent
        data_files_ok = True
        expected_files = [
             base_dir_path / "examples/payments/data/FAKE_PROD_BD_TH_FLAT_V3.csv",
             base_dir_path / "examples/payments/data/FAKE_ETS_D_CUST_PORTFOLIO.csv",
             base_dir_path / "examples/payments/data/rewriter_prompt.txt",
             base_dir_path / "examples/payments/data/code_prompt.txt",
        ]
        for f_path in expected_files:
             if not f_path.is_file():
                  logger.error(f"Required data/prompt file not found: {f_path}")
                  data_files_ok = False

        if data_files_ok:
             asyncio.run(main())
        else:
             logger.error("Please ensure all required data and prompt files exist.")