import unittest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock

# Add the project root to the path to allow imports from `dataqa`
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataqa.dbc.client import DBCClient
from dataqa.dbc.models import DBCRequest

# A minimal agent configuration dictionary for the test
# This mimics the structure of your YAML file
TEST_AGENT_CONFIG = {
    "agent_name": "dbc_test_agent",
    "use_case_name": "Test Case",
    "use_case_description": "A test case for DBC client.",
    "llm_configs": {
        "mock-llm": {
            "type": "dataqa.llm.base_llm.BaseLLM",
            "config": {"model": "mock-model"}
        }
    },
    "llm": {"default": "mock-llm"},
    "resource_manager_config": {
        "type": "dataqa.components.resource_manager.resource_manager.ResourceManager",
        "config": {"asset_directory": "/mock/assets"} # This path is mocked
    },
    "retriever_config": {
        "type": "dataqa.components.retriever.base_retriever.AllRetriever",
        "config": {
            "name": "all_retriever",
            "retrieval_method": "all",
            "resource_types": ["rule", "schema", "example"],
            "module_names": ["planner", "replanner", "retrieval_worker"]
        }
    },
    "workers": {
        "retrieval_worker": {
            "sql_execution_config": {
                "name": "mock_sql_executor",
                "backend": "duckdb",
                "data_files": [] # No data files needed as SQL is mocked
            }
        }
    }
}

# --- Mock Asset File Contents ---
# These strings simulate the content of your YAML asset files in S3.

MOCK_SCHEMA_YAML = """
tables:
  - table_name: FAKE_DATA
    description: A fake table with user data.
    columns:
      - name: user_id
        type: INTEGER
      - name: user_name
        type: TEXT
"""

MOCK_RULES_YAML = """
rules:
  - rule_name: test_rule
    module_name: planner
    instructions: "Always generate a simple plan."
"""

MOCK_EXAMPLES_YAML = """
examples:
  - query: "example query"
    module_name: "retrieval_worker"
    example:
      question: "What is a user name?"
      code: "SELECT user_name FROM FAKE_DATA LIMIT 1;"
      reasoning: "A simple example."
"""

# --- Main Test Class ---

class TestDBCClient(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up the mocks and the DBCClient instance for each test."""
        
        # 1. Mock the callables
        self.mock_llm_callable = AsyncMock()
        self.mock_s3_callable = MagicMock()
        self.mock_sql_callable = AsyncMock()

        # 2. Configure the behavior of the mocks
        self._configure_s3_mock()
        self._configure_llm_mock()
        self._configure_sql_mock()

        # 3. Instantiate the DBCClient with the mocks
        self.client = DBCClient(
            llm_callable=self.mock_llm_callable,
            s3_callable=self.mock_s3_callable,
            sql_callable=self.mock_sql_callable,
            agent_config=TEST_AGENT_CONFIG,
            asset_s3_prefix="s3://test-bucket/assets/",
            data_s3_prefix="s3://test-bucket/data/"
        )
        
        # Keep track of data written to S3
        self.s3_write_calls = []

    def _configure_s3_mock(self):
        """Configure the S3 mock to return asset files and track writes."""
        def s3_side_effect(s3_path, mode, content=None):
            print(f"Mock S3 called: path='{s3_path}', mode='{mode}'")
            if mode == 'r':
                if s3_path.endswith("schema.yml"):
                    return MOCK_SCHEMA_YAML.encode('utf-8')
                if s3_path.endswith("rules.yml"):
                    return MOCK_RULES_YAML.encode('utf-8')
                if s3_path.endswith("examples.yml"):
                    return MOCK_EXAMPLES_YAML.encode('utf-8')
                # For reading conversation history dataframes (not used in this test)
                return pd.DataFrame().to_parquet()
            elif mode == 'w':
                # Track what is being written to S3 to assert later
                self.s3_write_calls.append({'path': s3_path, 'content': content})
                return None
            
        self.mock_s3_callable.side_effect = s3_side_effect

    def _configure_llm_mock(self):
        """Configure a stateful LLM that returns different responses on each call."""
        
        # Response for the Planner step
        planner_response = """
        {
            "action": "continue",
            "plan": {
                "tasks": [
                    {
                        "worker": "retrieval_worker",
                        "task_description": "Find the name for user_id 123"
                    }
                ]
            }
        }
        """
        
        # Response for the Retrieval Worker's SQL Generator step
        sql_gen_response = """
        {
            "sql": "SELECT user_name FROM FAKE_DATA WHERE user_id = 123",
            "reasoning": "I need to select the user_name from the FAKE_DATA table.",
            "output": "user_name_df"
        }
        """

        # Response for the Replanner step, after the data is retrieved
        replanner_response = """
        {
            "action": "return",
            "response": {
                "response": "The user name has been found and is available in the dataframe 'user_name_df'.",
                "output_df_name": ["user_name_df"],
                "output_img_name": []
            }
        }
        """
        
        self.mock_llm_callable.side_effect = [
            planner_response,
            sql_gen_response,
            replanner_response,
        ]

    def _configure_sql_mock(self):
        """Configure the SQL mock to return a sample DataFrame."""
        mock_df = pd.DataFrame({"user_name": ["Alice"]})
        self.mock_sql_callable.return_value = mock_df

    async def test_process_query_end_to_end(self):
        """Test the full flow of processing a single query."""
        # 1. Create a sample request
        request = DBCRequest(
            user_query="What is the name for user 123?",
            conversation_id="conv-001",
            question_id="q-001"
        )

        # 2. Run the process_query method
        response = await self.client.process_query(request)

        # 3. Assert the results and interactions
        
        # Assert S3 reads for assets
        self.mock_s3_callable.assert_any_call("s3://test-bucket/assets/rules.yml", mode='r')
        self.mock_s3_callable.assert_any_call("s3://test-bucket/assets/schema.yml", mode='r')
        self.mock_s3_callable.assert_any_call("s3://test-bucket/assets/examples.yml", mode='r')

        # Assert LLM was called 3 times (Planner, SQL-Gen, Replanner)
        self.assertEqual(self.mock_llm_callable.call_count, 3)

        # Assert SQL executor was called with the correct SQL from the mock LLM
        self.mock_sql_callable.assert_awaited_once_with(
            sql="SELECT user_name FROM FAKE_DATA WHERE user_id = 123"
        )

        # Assert S3 write for the final dataframe
        self.assertEqual(len(self.s3_write_calls), 1)
        self.assertTrue(self.s3_write_calls[0]['path'].startswith("s3://test-bucket/data/dataframes/user_name_df"))
        
        # Assert the final response object
        self.assertIsNotNone(response)
        self.assertEqual(response.text, "The user name has been found and is available in the dataframe 'user_name_df'.")
        self.assertEqual(len(response.output_df_names), 1)
        self.assertTrue(response.output_df_names[0].startswith("s3://test-bucket/data/dataframes/user_name_df"))
        self.assertEqual(len(response.steps), 4) # retriever, planner, retrieval_worker, replanner

if __name__ == "__main__":
    unittest.main()