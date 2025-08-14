import argparse
import asyncio
import copy
import datetime
import os
import tempfile

import yaml

from benchmark.schema import BenchmarkConfig
from benchmark.test_pipeline import TestPipeline

# Define the predefined LLM configurations with names
predefined_configs = {
    "config_0": {
        "planner": "gpt-4o",
        "replanner": "gpt-4o",
        "retrieval_worker": "gpt-4o",
        "analytics_worker": "gpt-4o",
        "plot_worker": "gpt-4o",
    },
    "config_1": {
        "planner": "gpt-4.1",
        "replanner": "gpt-4.1",
        "retrieval_worker": "gpt-4.1",
        "analytics_worker": "gpt-4.1",
        "plot_worker": "gpt-4.1",
    },
    "config_2": {
        "planner": "o3-mini",
        "replanner": "o3-mini",
        "retrieval_worker": "o3-mini",
        "analytics_worker": "o3-mini",
        "plot_worker": "o3-mini",
    },
    "config_3": {
        "planner": "o3-mini",
        "replanner": "o3-mini",
        "retrieval_worker": "gpt-4o",
        "analytics_worker": "gpt-4o",
        "plot_worker": "gpt-4o",
    },
    "config_4": {
        "planner": "o3-mini",
        "replanner": "o3-mini",
        "retrieval_worker": "gpt-4.1",
        "analytics_worker": "gpt-4.1",
        "plot_worker": "gpt-4.1",
    },
}


# Load the configuration from a file
def load_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Save the modified configuration to a file
def save_config(config, file_path):
    with open(file_path, "w") as file:
        yaml.dump(config, file)


# Update the LLM configuration for each use case
def update_config(main_config, llm_config):
    main_config = copy.deepcopy(main_config)
    temp_cwd_configs = []
    for use_case in main_config["use_case_config"]:
        cwd_config_path = use_case["cwd_config"]

        # Load the original cwd configuration
        cwd_config = load_config(cwd_config_path)

        # Update the LLM configuration
        for role, model in llm_config.items():
            cwd_config["llm"][role] = model

        # Save the modified cwd configuration to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".yaml", mode="w"
        ) as temp_file:
            save_config(cwd_config, temp_file.name)
            temp_cwd_configs.append(temp_file.name)

        # Update the use case to point to the temporary config
        use_case["cwd_config"] = temp_file.name

    return main_config, temp_cwd_configs


# Run the benchmarking script
async def run_benchmarking(config):
    test_config = BenchmarkConfig(**config)
    test_pipeline = TestPipeline(config=test_config)
    await test_pipeline.run()


def get_args():
    parser = argparse.ArgumentParser(description="Benchmarking Script")
    parser.add_argument(
        "-a",
        "--agent_config",
        type=str,
        required=True,
        help="Path to the agent configuration file",
    )
    return parser.parse_args()


def main():
    # Get command line arguments
    args = get_args()

    # Path to the main agent configuration file
    main_config_path = args.agent_config

    # Load the main configuration
    main_config = load_config(main_config_path)

    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Check for environment variables
    if not os.environ.get("CERT_PATH"):
        os.environ["CERT_PATH"] = input("Path to PEM=")
    if not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = input("OPENAI API BASE=")

    # Iterate over each predefined LLM configuration
    for config_name, llm_config in predefined_configs.items():
        # Update the configuration with the current LLM configuration
        modified_main_config, temp_cwd_configs = update_config(
            main_config, llm_config
        )

        # Modify output and log paths to include the configuration name and current date
        modified_main_config["output"] = (
            f"benchmark/output/{config_name}_agent_{current_date}_run_{main_config['run_id']}"
        )
        modified_main_config["log"] = (
            f"benchmark/log/{config_name}_agent_{current_date}_run_{main_config['run_id']}.log"
        )

        # Create directories if they don't exist
        os.makedirs(modified_main_config["output"], exist_ok=True)
        os.makedirs(os.path.dirname(modified_main_config["log"]), exist_ok=True)

        # Run the benchmarking with the modified configuration
        print(
            f"Running benchmark with LLM configuration '{config_name}': {llm_config}"
        )
        asyncio.run(run_benchmarking(modified_main_config))

        # Optionally, remove the temporary configuration files
        for temp_cwd_config in temp_cwd_configs:
            os.remove(temp_cwd_config)


if __name__ == "__main__":
    main()
