import argparse
import asyncio
import os

import yaml

from dataqa.benchmark.schema import BenchmarkConfig
from dataqa.benchmark.test_pipeline import TestPipeline


def get_args():
    parser = argparse.ArgumentParser(description="CWD Benchmark")
    parser.add_argument(
        "-c", "--config", type=str, help="path to benchmark config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not os.environ.get("CERT_PATH"):
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = input("AZURE_OPENAI_API_KEY=")
        if not os.environ.get("AZURE_ENDPOINT", ""):
            os.environ["AZURE_ENDPOINT"] = input("OPENAI_API_BASE=")

    if os.path.isfile(args.config):
        test_config_data = yaml.safe_load(open(args.config))
    else:
        raise f"Config file {args.config} doesn't exist."

    test_config = BenchmarkConfig(**test_config_data)

    test_pipeline = TestPipeline(config=test_config)

    asyncio.run(test_pipeline.run())
