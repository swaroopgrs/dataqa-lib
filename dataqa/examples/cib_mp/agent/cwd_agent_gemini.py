import asyncio
import os
from pathlib import Path

from dataqa.agent.cwd_agent.cwd_agent import CWDAgent, CWDState
from dataqa.memory import Memory
from dataqa.utils.agent_util import AgentResponseParser
from dataqa.utils.langgraph_utils import (
    CONFIGURABLE,
    DEFAULT_THREAD,
    THREAD_ID,
)

SCRIPT_DIR = Path(__file__).resolve().parent

def run_agent(query):
    memory = Memory()

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set!")
    
    config_path = SCRIPT_DIR / "cwd_agent_prompt_template_gemini.yaml"

    agent = CWDAgent.from_config_path(str(config_path), memory)

    config = {
        CONFIGURABLE: {
            THREAD_ID: DEFAULT_THREAD,
        }
    }

    state = CWDState(query=query)
    
    # asyncio.run should be in the main execution block
    state, all_events = asyncio.run(agent(state, config))
    
    agent_response_parser = AgentResponseParser(all_events, memory, config)
    agent_response_parser.pretty_print_output()
    return state, all_events, agent_response_parser

if __name__ == "__main__":
    example_questions = [
        "what is the co_id for td id 881",
        "what is the market segment for co_id 1003",
        "What is the total gross sales volume and units for 1004 co_id for the date of 18th March 2025?",
        "What is the trend of gross sales volume for co_id 1003 over the past quarter?",
        "Plot the daily gross sales volume for co_id 1005 during the second week of April 2025",
    ]
    query = "What is the total gross sales volume by MOP code for co_id 1001 for Q12025 for Visa?"
    run_agent(query)