import streamlit as st
import asyncio
import os
import sys
from pathlib import Path
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataqa.core.client import CoreRequest
from dataqa.integrations.local.client import LocalClient

def show():
    st.markdown("### Test your CWD Agent with interactive queries")

    # Check if project is loaded
    if 'current_project' not in st.session_state or st.session_state.current_project is None:
        st.warning("Please select a project in the Project Manager first.")
        return

    project_path = st.session_state.current_project
    agent_config_path = project_path / "agent_config.yml"

    if not agent_config_path.exists():
        st.error(f"Agent configuration not found: {agent_config_path}")
        return

    # Environment setup
    st.subheader("Environment Setup")

    col1, col2 = st.columns(2)

    with col1:
        api_key = st.text_input("Azure OpenAI API Key", type="password",
                              value=os.environ.get("AZURE_OPENAI_API_KEY", ""))
        if api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = api_key

    with col2:
        api_base = st.text_input("OpenAI API Base",
                               value=os.environ.get("OPENAI_API_BASE", ""))
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base

    # Agent configuration display
    with st.expander("Agent Configuration", expanded=False):
        try:
            with open(agent_config_path, 'r') as f:
                config_content = f.read()
            st.code(config_content, language='yaml')
        except Exception as e:
            st.error(f"Error reading agent config: {str(e)}")

    # Query interface
    st.subheader("Query Interface")

    # Conversation History
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Query input
    query = st.text_area("Enter your query", height=100,
                         placeholder="e.g., What is the total sales for company 1001?")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🚀 Run Query", disabled=not query):
            if not api_key or not api_base:
                st.error("Please provide Azure OpenAI credentials")
            else:
                run_agent_query(agent_config_path, query)
    with col2:
        if st.button("🧹 Clear History"):
            st.session_state.conversation_history = []
            st.rerun()

    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for i, turn in enumerate(st.session_state.conversation_history):
            with st.expander(f"Query ({i+1}): {turn['query']}", expanded=(i == len(st.session_state.conversation_history) - 1)):
                if 'error' in turn:
                    st.error(f"Error: {turn['error']}")
                if 'response' in turn:
                    st.markdown(f"**Response:** {turn['response']}")
                if 'dataframes' in turn:
                    for j, df in enumerate(turn['dataframes']):
                        st.markdown(f"**DataFrame ({j+1}):**")
                        st.dataframe(df)
                # Display images
                if 'images' in turn:
                    st.markdown("**Generated Images:**")
                    for j, img_bytes in enumerate(turn['images']):
                        st.image(img_bytes, caption=f"Generated Image {j+1}")

                # Display execution steps
                if 'steps' in turn:
                    with st.expander("Execution Steps"):
                        for step in turn['steps']:
                            st.markdown(f"**{step.name}**")
                            st.text(step.content)

def run_agent_query(config_path: Path, query: str):
    """Run the agent query and display results"""
    with st.spinner("Running agent query..."):
        try:
            # Create client
            client = LocalClient(config_path=str(config_path))

            # Create request
            request = CoreRequest(
                user_query=query,
                question_id=f"ui_query_{len(st.session_state.conversation_history) + 1}",
                history=st.session_state.conversation_history,
                output_test= turn.get("response", "") if (turn := st.session_state.conversation_history) else ""
            )
            # Run the query
            response = asyncio.run(process_query_async(client, request))

            # Store in conversation history
            turn = {
                'query': query,
                'response': response.text,
                'dataframes': response.output_dataframes,
                'images': response.output_images,
                'steps': response.steps
            }
            st.session_state.conversation_history.append(turn)
            st.success("Query completed successfully!")
            st.rerun()
        except Exception as e:
            error_msg = f"Error running query: {str(e)}\n\n{traceback.format_exc()}"
            st.error(error_msg)
            # Store error in history
            turn = {
                'query': query,
                'error': str(e)
            }
            st.session_state.conversation_history.append(turn)

async def process_query_async(client, request):
    """Async wrapper for processing query"""
    response = None
    async for chunk in client.process_query(request, streaming=True, summarize=True):
        response = chunk # This is the final CoreResponse
        break
    return response