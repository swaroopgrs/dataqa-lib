# examples/cib_mp/agent/run_dbc_client.py
import asyncio
import os
import uuid
from pathlib import Path
from typing import Set, Any, List
import pandas as pd
import yaml

# --- Library Imports ---
# Import the DBC client and its required models
from dataqa.integrations.dbc.client import DBCClient
from dataqa.integrations.dbc.models import DBCRequest, DBCResponse, UsecaseConfig, FileType, IngestionData

# Import core library components needed for mocking
from dataqa.core.llm.openai import AzureOpenAI, AzureOpenAIConfig
from dataqa.core.components.code_executor.in_memory_code_executor import InMemoryCodeExecutor
from dataqa.core.data_models.asset_models import Rules, DatabaseSchema, Examples
from langchain_core.messages import BaseMessage

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
CIB_MP_PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = CIB_MP_PROJECT_DIR / "data"

# --- Mock Implementations ---
# These functions and classes simulate the environment provided by the DBC service.

def mock_asset_callable(config_id: uuid.UUID, file_types: Set[FileType]) -> IngestionData:
    """
    Simulates fetching asset files from a remote source (like S3).
    For this mock, it reads them from the local file system.
    """
    print(f" MOCK [asset_callable]: Called for config_id={config_id} with types={file_types}")
    
    ingestion_data = {}
    if FileType.RULES in file_types:
        rules_path = DATA_DIR / "rules.yml"
        ingestion_data['rules'] = Rules(**yaml.safe_load(open(rules_path)))
    if FileType.SCHEMA in file_types:
        schema_path = DATA_DIR / "schema.yml"
        ingestion_data['schema'] = DatabaseSchema(**yaml.safe_load(open(schema_path)))
    if FileType.EXAMPLES in file_types:
        examples_path = DATA_DIR / "examples.yml"
        ingestion_data['examples'] = Examples(**yaml.safe_load(open(examples_path)))
        
    return IngestionData(**ingestion_data)

def mock_storage_callable(data: bytes, path_suffix: str) -> str:
    """Simulates writing data to a persistent store and returning its path."""
    fake_s3_path = f"s3://mock-dataqa-bucket/{uuid.uuid4()}/{path_suffix}"
    print(f" MOCK [storage_callable]: 'Saving' {len(data)} bytes to {fake_s3_path}")
    # For actual testing, you might want to save the file locally to inspect it
    # output_dir = SCRIPT_DIR / "output" / "dbc_mock_storage"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # with open(output_dir / path_suffix.replace('/', '_'), 'wb') as f:
    #     f.write(data)
    return fake_s3_path

class MockLLMService:
    """A simplified mock of the DBC LLMService to provide the invocation function."""
    def __init__(self):
        # Use a real LLM client as the backend for our mock service
        # Ensure the model supports tool calling, like gpt-4o.
        print(" MOCK [LLM Service]: Initializing real AzureOpenAI client for mock...")
        self.primary_llm = AzureOpenAI(
            AzureOpenAIConfig(
                model="gpt-4o-2024-08-06",
                api_version="2024-08-01-preview",
                api_type="azure_ad",
                temperature=0,
                base_url=os.environ["OPENAI_API_BASE"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
            )
        )

    async def llm_invoke_with_retries(self, model: str, messages: List[BaseMessage]) -> BaseMessage:
        """
        This function signature exactly matches the one provided by the DBC service.
        Our mock simply calls the primary LLM directly, simulating a successful call.
        """
        print(f" MOCK [llm_invoke_with_retries]: Invoking model '{model}' with {len(messages)} messages...")
        # A real implementation would have complex retry/fallback logic here.
        return await self.primary_llm.ainvoke(messages)

class MockSQLService:
    """A class to hold the stateful SQL Executor for the mock SQL service."""
    def __init__(self):
        self._sql_executor = None

    def get_sql_executor(self):
        if self._sql_executor is None:
            print(" MOCK [SQL Service]: Initializing InMemoryCodeExecutor (DuckDB) for mock...")
            config = {
                "name": "mock_sql_executor",
                "data_files": [
                    {"path": str(DATA_DIR / "FAKE_PROD_BD_TH_FLAT_V3.csv"), "table_name": "PROD_BD_TH_FLAT_V3"},
                    {"path": str(DATA_DIR / "FAKE_ETS_D_CUST_PORTFOLIO.csv"), "table_name": "EIS_D_CUST_PORTFOLIO"},
                ]
            }
            self._sql_executor = InMemoryCodeExecutor(config)
        return self._sql_executor

    async def sql_callable(self, config_id: uuid.UUID, sql_query: str) -> pd.DataFrame:
        """Async mock for the SQL callable, using a local in-memory DB."""
        print(f" MOCK [sql_callable]: Executing SQL for config_id={config_id}")
        executor = self.get_sql_executor()
        from pydantic import create_model
        SqlInput = create_model('SqlInput', code=(str, ...))
        result = await executor.run(SqlInput(code=sql_query))
        if result.error:
            raise Exception(f"SQL Execution Error: {result.error}")
        return pd.read_json(result.dataframe[0])

# --- Main Runner ---
async def main():
    print("🚀 Initializing Mocks for DBCClient End-to-End Test...")
    mock_llm_service = MockLLMService()
    mock_sql_service = MockSQLService()

    # 1. Define the UsecaseConfig (this would be provided by the DBC service)
    usecase_config = UsecaseConfig(
        config_id=uuid.uuid4(),
        tenant_id="mock_tenant",
        usecase_name="Merchant Payments (DBC Mock)",
        usecase_description="This is a mocked run of the CIB Merchant Payments use case via the DBC interface."
    )

    # 2. Define the DBCRequest (this would be sent by the DBC service)
    query = "Plot the daily gross sales volume for co_id 1005 during the second week of April 2025"
    request = DBCRequest(
        user_query=query,
        conversation_id="dbc_test_session_plot_01",
        question_id=str(uuid.uuid4())
    )

    # 3. Instantiate the DBCClient with the mock callables and the request/config objects
    client = DBCClient(
        usecase_config=usecase_config,
        request=request,
        llm_callable=mock_llm_service.llm_invoke_with_retries,
        asset_callable=mock_asset_callable,
        sql_callable=mock_sql_service.sql_callable,
        storage_callable=mock_storage_callable
    )
    
    print(f"\n▶️  Processing DBCRequest with query: '{query}'")

    # 4. Process the request using the client's main method
    response: DBCResponse = await client.process_query()

    # 5. Print the structured results from the DBCResponse
    print("\n" + "="*20 + " DBC RESPONSE " + "="*20)
    print("\n📝 Final Text Response:")
    print(response.text)

    if response.output_df_names:
        print("\n📊 Output DataFrame S3 Paths:")
        for path in response.output_df_names:
            print(f"   - {path}")

    if response.output_image_names:
        print("\n🖼️ Output Image S3 Paths:")
        for path in response.output_image_names:
            print(f"   - {path}")

    print("\n" + "="*20 + " DEBUG INFO " + "="*20)
    print("\n⚙️ Agent Execution Steps:")
    for step in response.steps:
        print(f"\n--- {step.name} ---")
        print(step.content)

    print("\n✅ DBC Client test finished.")

if __name__ == "__main__":
    # Ensure necessary environment variables are set for the mock LLM
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        raise ValueError("Please set the AZURE_OPENAI_API_KEY environment variable.")
    if not os.environ.get("OPENAI_API_BASE"):
        raise ValueError("Please set the OPENAI_API_BASE environment variable.")
    
    asyncio.run(main())