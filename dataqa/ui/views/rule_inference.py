import asyncio
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataqa.core.components.knowledge_extraction.rule_inference_batch_test import (
    RuleInferenceExperiment,
)


def show():
    st.header("📜 Rule Inference")
    st.markdown(
        "*Automatically infer business rules by comparing generated and expected SQL queries*"
    )

    # Check if project is loaded
    if (
        "current_project" not in st.session_state
        or st.session_state.current_project is None
    ):
        st.warning("Please select a project in the Project Manager first.")
        return

    project_path = st.session_state.current_project
    rule_config_path = project_path / "rule_inference_config.yml"

    # Configuration section
    st.subheader("Rule Inference Configuration")
    try:
        if rule_config_path.exists():
            with open(rule_config_path, "r") as f:
                config = yaml.safe_load(f)

            # Display and edit configuration
            with st.expander("Configuration", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    config["max_iteration"] = st.number_input(
                        "Max Iterations",
                        min_value=1,
                        max_value=10,
                        value=config.get("max_iteration", 3),
                        help="Maximum number of rule inference iterations per question",
                    )
                    config["multi-tenant_subscription"] = st.checkbox(
                        "Multi-tenant Subscription",
                        value=config.get("multi-tenant_subscription", False),
                        help="Use multi-tenant Azure subscription",
                    )
                with col2:
                    # Update paths to be relative to project
                    config["original_config_file"] = str(
                        project_path / "agent_config.yml"
                    )
                    config["test_data_file"] = str(
                        project_path / "data" / "evaluation_data.yml"
                    )
                    config["output_file_path"] = str(
                        project_path / "output" / "rule_inference"
                    )

                # Save configuration
                if st.button("💾 Save Configuration"):
                    with open(rule_config_path, "w") as f:
                        yaml.dump(
                            config, f, default_flow_style=False, sort_keys=False
                        )
                    st.success("Configuration saved!")

                # Show full config
                with st.expander("Full Configuration (YAML)"):
                    st.code(
                        yaml.dump(config, default_flow_style=False),
                        language="yaml",
                    )
        else:
            st.error(
                f"Rule inference configuration not found: {rule_config_path}"
            )
            return
    except Exception as e:
        st.error(f"Error reading rule inference config: {str(e)}")
        return

    # Test data validation
    st.subheader("Test Data Validation")
    test_data_path = Path(config.get("test_data_file"))
    try:
        if test_data_path.exists():
            with open(test_data_path, "r") as f:
                test_data = yaml.safe_load(f)
            valid_items = []
            for item in test_data.get("data", []):
                if (
                    item.get("active", True)
                    and item.get("solution")
                    and item.get("question")
                    and item.get("component_groundtruth", {}).get(
                        "ground_truth_output"
                    )
                ):
                    valid_items.append(item)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Test Cases", len(test_data.get("data", [])))
            with col2:
                st.metric("Valid for Rule Inference", len(valid_items))
            with col3:
                st.metric(
                    "Invalid Cases",
                    len(test_data.get("data", [])) - len(valid_items),
                )

            if valid_items:
                # Show test cases
                with st.expander("Valid Test Cases"):
                    df_data = []
                    for item in valid_items:
                        df_data.append(
                            {
                                "ID": item.get("id", ""),
                                "Question": item.get("question", "")[:100]
                                + "..."
                                if len(item.get("question", "")) > 100
                                else "",
                                "Has SQL": bool(
                                    item.get("solution", [{}])[0]
                                    .get("function_arguments", {})
                                    .get("sql", "")
                                ),
                                "Has Ground Truth": bool(
                                    item.get("component_groundtruth", {}).get(
                                        "ground_truth_output"
                                    )
                                ),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(df_data), use_container_width=True
                    )

                # Test case selection
                st.subheader("Test Case Selection")
                mode = st.radio(
                    "Select test cases to process:",
                    ["All valid cases", "Specific cases", "First N cases"],
                )

                selected_ids = None
                if mode == "Specific cases":
                    available_ids = [item["id"] for item in valid_items]
                    selected_ids = st.multiselect(
                        "Select test case IDs", available_ids
                    )
                elif mode == "First N cases":
                    n_cases = st.number_input(
                        "Number of cases",
                        min_value=1,
                        max_value=len(valid_items),
                        value=min(5, len(valid_items)),
                    )
                    selected_ids = [
                        item["id"] for item in valid_items[:n_cases]
                    ]

                st.info(
                    f"Selected {len(selected_ids or [])} test cases: {', '.join(selected_ids or [])}"
                )
            else:
                st.warning("No valid test cases found for rule inference")
                return
        else:
            st.error(f"Test data file not found: {test_data_path}")
            return
    except Exception as e:
        st.error(f"Error reading test data: {str(e)}")
        return

    # Environment setup
    st.subheader("Environment Setup")
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input(
            "Azure OpenAI API Key",
            type="password",
            value=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        )
        if api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
    with col2:
        api_base = st.text_input(
            "OpenAI API Base", value=os.environ.get("OPENAI_API_BASE", "")
        )
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base

        if config.get("multi_tenant_subscription"):
            cert_path = st.text_input(
                "Certificate Path", value=os.environ.get("CERT_PATH", "")
            )
            if cert_path:
                os.environ["CERT_PATH"] = cert_path

    # Run rule inference
    st.subheader("Run Rule Inference")
    inference_mode = st.radio(
        "Inference Mode",
        [
            "New Questions (Individual)",
            "Update Rules (Batch)",
            "Consolidate Rules",
        ],
    )

    if st.button(
        "🚀 Start Rule Inference", disabled=not (api_key and api_base)
    ):
        if inference_mode == "New Questions (Individual)":
            run_rule_inference(config, selected_ids, "individual")
        elif inference_mode == "Update Rules (Batch)":
            run_rule_inference(config, selected_ids, "batch")
        else:  # Consolidate Rules
            run_rule_inference(config, selected_ids, "consolidate")


def run_rule_inference(config: dict, selected_ids: list, mode: str):
    """Run the rule inference experiment"""
    with st.spinner(f"Running rule inference ({mode})..."):
        try:
            # Create experiment
            experiment = RuleInferenceExperiment(
                config_path=Path(config["original_config_file"]),
                test_data_file=Path(config["test_data_file"]),
                output_file_path=Path(config["output_file_path"]),
                max_iteration=config["max_iteration"],
                multi_tenant_subscription=config.get(
                    "multi_tenant_subscription", False
                ),
            )

            # Load test data
            experiment.load_test_data(selected_ids)

            # Run based on mode
            if mode == "individual":
                asyncio.run(experiment.run_question_tuning_batch())
                st.success("✅ Question tuning completed!")
            elif mode == "batch":
                asyncio.run(experiment.update_rules_from_question_batch())
                st.success("✅ Batch rule update completed!")
            elif mode == "consolidate":
                if (
                    hasattr(experiment, "experiment_result")
                    and experiment.experiment_result
                ):
                    combined_rules = asyncio.run(experiment.consolidate_rules())
                    st.success("✅ Rule consolidation completed!")
                else:
                    st.error(
                        "No experiment results found. Run tune questions or update rules first."
                    )
                    return

            # Show consolidated rules
            if combined_rules:
                st.subheader("Consolidated Rules")
                st.code(combined_rules, language="text")

                # Option to save to rules.yml
                if st.button("Save to rules.yml"):
                    save_consolidated_rules(
                        combined_rules, config["config_path"]
                    )
            else:
                st.error(
                    "No experiment results found. Run tune questions or update rules first."
                )
            st.rerun()
        except Exception as e:
            st.error(f"Error running rule inference: {str(e)}")
            import traceback

            st.code(traceback.format_exc())


def show_rule_inference_results(results_dir: Path):
    """Display rule inference results"""
    # Look for result files
    pickle_files = list(results_dir.glob("rule_inference.xlsx"))
    if pickle_files:
        pickle_files = list(results_dir.glob("rule_inference.pkt"))

    st.subheader("Excel Results")
    for result_file in pickle_files:
        with st.expander(f"{result_file.name}"):
            try:
                df = pd.read_excel(result_file)

                # Summary metrics
                if "llm_label" in df.columns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        correct_count = (df["llm_label"] == "correct").sum()
                        st.metric(
                            "Correct",
                            f"{correct_count} ({correct_count / len(df):.2%})",
                        )
                    with col2:
                        wrong_count = (df["llm_label"] == "wrong").sum()
                        st.metric(
                            "Wrong",
                            f"{wrong_count} ({wrong_count / len(df):.2%})",
                        )
                    with col3:
                        if "iteration_count" in df.columns:
                            avg_iterations = (
                                df["iteration_count"].mean()
                                if len(df.columns)
                                else 0
                            )
                            st.metric("Avg Iterations", f"{avg_iterations:.2f}")

                # Show data
                st.dataframe(df, use_container_width=True)

                # Download button
                with open(result_file, "rb") as f:
                    st.download_button(
                        label=f"Download {result_file.name}",
                        data=f,
                        file_name=result_file.name,
                    )
            except:
                pass

    st.subheader("Detailed Results (Pickle)")
    for pickle_file in pickle_files:
        with st.expander(f"{pickle_file.name}"):
            try:
                with open(pickle_file, "rb") as f:
                    results = pickle.load(f)

                st.info(f"Contains {len(results)} experiment results")

                # Show sample results
                if results:
                    st.subheader("Sample Result Structure")
                    sample_result = results[0]
                    if len(sample_result) > 0:
                        st.text(f"Question: {sample_result[0]}")
                        st.text(f"Iterations: {sample_result[1]}")
                        st.text(f"Has Rules: {sample_result[2]}")

                    # Show extracted rules if available
                    if len(results) > 0 and len(results[0]) > 0:
                        with st.expander("Sample Extracted Rules"):
                            rules = sample_result[0]
                            if hasattr(rules, "rules") and hasattr(
                                rules.rules[0], "rules"
                            ):
                                for i, rule in enumerate(rules.rules[0].rules):
                                    st.text(f"{i + 1}. {rule}")
            except Exception as e:
                st.error(f"Error reading pickle file: {str(e)}")

    if not result_files and not pickle_files:
        st.info(
            "No rule inference results found. Run rule inference to see results here."
        )


def save_consolidated_rules(rules_text: str, project_path: str):
    """Save consolidated rules to rules.yml"""
    try:
        project_path = Path(project_path)
        rules_file_path = project_path / "data" / "rules.yml"

        # Parse rules text into structured format
        rules_list = []
        for line in rules_text.split("\n"):
            if line.strip():
                line_text = line.split(". ")[1].strip()
                rules_list.append(
                    {
                        "rule_name": f"inferred_rule_{len(rules_list) + 1}",
                        "module_name": "planner",
                        "instructions": line_text,
                        "tags": ["inferred"],
                        "search_content": "",
                    }
                )

        # Load existing rules if any
        existing_rules = []
        if rules_file_path.exists():
            with open(rules_file_path, "r") as f:
                existing_data = yaml.safe_load(f)
                existing_rules = existing_data.get("rules", [])

        # Combine and save rules
        all_rules = existing_rules + rules_list
        with open(rules_file_path, "w") as f:
            yaml.dump(
                {"rules": all_rules},
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        st.success(
            f"Saved {len(rules_list)} consolidated rules to {rules_file_path}"
        )
    except Exception as e:
        st.error(f"Error saving rules: {str(e)}")
