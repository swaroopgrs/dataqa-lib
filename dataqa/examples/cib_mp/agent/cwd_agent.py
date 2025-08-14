import asyncio
import os
from pathlib import Path

from dataqa.agent.cwd_agent.cwd_agent import CWDAgent, CWDState
from dataqa.memory import Memory
from dataqa.utils.agent_util import AgentResponseParser
from dataqa.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    DEFAULT_THREAD,
    THREAD_ID,
)

SCRIPT_DIR = Path(__file__).resolve().parent

def run_agent(query):
    memory = Memory()

    base_url = os.environ.get("OPENAI_API_BASE", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError("API key is missing")
    if not base_url:
        raise ValueError("Base URL is missing")
    
    config_path = SCRIPT_DIR / "cwd_agent_prompt_template.yaml"

    agent = CWDAgent.from_config_path(str(config_path), memory)

    state = CWDState(query=query)
    config = {
        CONFIGURABLE: {
            THREAD_ID: DEFAULT_THREAD,
            API_KEY: api_key,
            BASE_URL: base_url,
        }
    }

    # run agent
    state, all_events = asyncio.run(agent(state, config))
    agent_response_parser = AgentResponseParser(all_events, memory, config)
    agent_response_parser.pretty_print_output()
    return state, all_events, agent_response_parser


if __name__ == "__main__":
    example_questions = [
        "what is the co_id for td id 881",
        "what is the market segment for co_id 1003",
        "what is the company name for td 666",
        "what is the mcc code associated with td 448",
        "which country does the td 100 belong?",
        "which state does the TD 666 belong to?",
        "what is the name of the TD 881",
        "what is the cust id for TD 881",
        "what is the cust key for TD 568",
        "what is the ecid associated with TD 619",
        "what companies are associated with ecid 3219824?",
        "what is the list of active tds in co_id 1005",
        "what unique mcc are covered under co id 1002?",
        "give me a count of tds which are having different status in co_id 1004",
        "are multiple cust keys associated with the td_id?",
        "what is the list of cust_key and td_id associated with the co_id 1001? along with td name and td region",
        "what is the list of cust_key and td_id associated with the co_id 1001? along with td name and td region is us",
        "What is the total gross sales volume and units for 1004 co_id for the date of 18th March 2025?",
        "What is the total gross sales volume and units for 718 td_id for the date of 20th Feb 2025?",
        "What is the sales volume for 1005 co_id for the second week of April 2025?",
        "What is the sales volume for 121 td_id for the second week of April 2025?",
        "What is the total gross sales volume and units for 1003 co_id for the month of April 2025?",
        "What is the total gross sales volume and units for 121 td_id for the month of September 2024?",
        "What is the total gross sales volume and units for 1001 co_id for the Q1 of 2025?",
        "What is the total gross sales volume and units for 121 td_id for the Q1 of 2025?",
        "What is the total gross sales volume by MOP code for co_id 1003 for the month of Jan 2025?",
        "What is the total gross sales volume by MOP code for co_id 1001 for Q12025 for Visa?",
        "What is the trend of gross sales volume for co_id 1003 over the past quarter?",
        "Plot the daily gross sales volume for co_id 1005 during the second week of April 2025",
    ]
    query = "What is the total gross sales volume by MOP code for co_id 1001 for Q12025 for Visa?"
    state, all_events, agent_response_parser = run_agent(query)

