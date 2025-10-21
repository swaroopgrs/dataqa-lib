import streamlit as st
import sys
from pathlib import Path

# Add the parent directory to the path so we can import dataqa modules
sys.path.append(str(Path(__file__).parent.parent))

from dataqa.ui.views import agent_playground
from views import evaluation, data_scanner, rule_inference, project_manager

st.set_page_config(
    page_title="DataQA Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 DataQA Suite")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "🏠 Project Manager": project_manager,
        "🤖 Agent Playground": agent_playground,
        "📊 Evaluation": evaluation,
        "🔍 Data Scanner": data_scanner,
        "🧠 Rule Inference": rule_inference,
    }

    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    # Display the selected page
    pages[selected_page].show()

if __name__ == "__main__":
    main()
