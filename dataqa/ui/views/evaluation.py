import streamlit as st
import asyncio
import pandas as pd
from pathlib import Path
import sys
import os
import subprocess
import threading
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataqa.benchmark.schema import BenchmarkConfig
from dataqa.benchmark.test_pipeline import TestPipeline

def show():
    st.markdown("### 🧪 Evaluation")
    st.markdown("*Evaluate your agent's performance using benchmark datasets*")

    # Check if project is loaded
    if 'current_project' not in st.session_state or st.session_state.current_project is None:
        st.warning("Please select a project in the Project Manager first.")
        return

    project_path = st.session_state.current_project
    eval_config_path = project_path / "evaluation_config.yml"

    # Configuration section
    st.sidebar.subheader("Evaluation Configuration")
    if eval_config_path.exists():
        with st.sidebar.expander("Current Configuration", expanded=True):
            try:
                with open(eval_config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Display and allow editing of key parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    config['num_run'] = st.number_input("Number of Runs",
                                                        min_value=1, max_value=10,
                                                        value=config.get('num_run', 1))
                    config['batch_size'] = st.number_input("Batch Size",
                                                         min_value=1, max_value=20,
                                                         value=config.get('batch_size', 4))
                    config['run_prediction'] = st.checkbox("Run Prediction",
                                                        value=config.get('run_prediction', True))
                
                with col2:
                    config['run_llm_eval'] = st.checkbox("Run LLM Evaluation",
                                                      value=config.get('run_llm_eval', True))
                    config['resume'] = st.checkbox("Resume Previous Run",
                                                 value=config.get('resume', False))
                    config['debug'] = st.checkbox("Debug Mode",
                                                  value=config.get('debug', False))

                # Update output paths to be relative to project
                config['output'] = str(project_path / "output" / "evaluation")
                config['log'] = str(project_path / "output" / "evaluation.log")

                # Save updated config
                if st.button("💾 Save Configuration"):
                    with open(eval_config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    st.success("Configuration saved!")

                # Show full config
                with st.expander("Full Configuration"):
                    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language='yaml')
            except Exception as e:
                st.error(f"Error reading evaluation config: {str(e)}")
                return
    else:
        st.error(f"Evaluation configuration not found: {eval_config_path}")
        return

    # Test data preview
    st.subheader("Test Data")
    eval_data_path = project_path / "data" / "evaluation_data.yml"

    if eval_data_path.exists():
        try:
            with open(eval_data_path, 'r') as f:
                eval_data = yaml.safe_load(f)
            
            # Show test data summary
            test_items = eval_data.get('data', [])
            active_items = [item for item in test_items if item.get('active', True)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Test Cases", len(test_items))
            with col2:
                st.metric("Active Test Cases", len(active_items))
            with col3:
                st.metric("Inactive Test Cases", len(test_items) - len(active_items))
            
            # Show test cases in a table
            if test_items:
                df_data = []
                for item in test_items:
                    df_data.append({
                        'ID': item.get('id', ''),
                        'Question': item.get('question', '')[:100] + '...' if len(item.get('question', '')) > 100 else '',
                        'Active': item.get('active', True),
                        'Label': ', '.join(item.get('labels', []))
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Test case details
                selected_id = st.selectbox("Select test case for details",
                                           [item['id'] for item in test_items])
                
                if selected_id:
                    selected_item = next(item for item in test_items if item['id'] == selected_id)
                    with st.expander(f"Details for {selected_id}", expanded=True):
                        st.markdown(f"**Question:** {selected_item.get('question', '')}")
                        st.markdown(f"**Ground Truth Output:**")
                        st.code(selected_item.get('ground_truth_output', {}).get('language', 'text'))
                        st.markdown(f"**Solution:**")
                        sql = selected_item['solution'][0].get('function_arguments', {}).get('sql', '')
                        st.code(sql, language='sql')
        except Exception as e:
            st.error(f"Error reading evaluation data: {str(e)}")
    else:
        st.error(f"Evaluation data not found: {eval_data_path}")
    
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
            
    # Run evaluation
    st.subheader("Run Evaluation")
    if st.button("🚀 Start Evaluation", disabled=not (api_key and api_base)):
        run_evaluation(eval_config_path, config)
        
    # Results section
    st.subheader("Evaluation Results")
    results_dir = project_path / "output" / "evaluation"
    if results_dir.exists():
        show_evaluation_results(results_dir)
    else:
        st.info("No evaluation results found. Run an evaluation to see results here.")
        
def run_evaluation(config_path: Path, config: dict):
    """Run the evaluation pipeline"""
    
    # Initialize session state for tracking
    if 'evaluation_running' not in st.session_state:
        st.session_state.evaluation_running = False
    
    if st.session_state.evaluation_running:
        st.warning("Evaluation is already running!")
        return
        
    st.session_state.evaluation_running = True
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()
    
    try:
        # Create benchmark config
        benchmark_config = BenchmarkConfig(**config)
        # Create test pipeline
        test_pipeline = TestPipeline(config=benchmark_config)
        
        status_text.text("Starting evaluation pipeline...")
        
        # Run evaluation in a separate thread to avoid blocking UI
        def run_pipeline():
            try:
                asyncio.run(test_pipeline.run())
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
            finally:
                st.session_state.evaluation_running = False
                
        # Start the pipeline
        thread = threading.Thread(target=run_pipeline)
        thread.start()
        
        # Monitor progress
        while thread.is_alive():
            progress_bar.progress(0.5) # Indeterminate progress
            status_text.text("Evaluation running... Check logs for details.")
            time.sleep(1)
            
        progress_bar.progress(1.0)
        status_text.text("Evaluation completed!")
        st.success("Evaluation finished successfully")
        
    except Exception as e:
        st.error(f"Error running evaluation: {str(e)}")
    finally:
        st.session_state.evaluation_running = False
        
def show_evaluation_results(results_dir: Path):
    """Display evaluation results"""
    
    # Look for evaluation.yml (summary results)
    eval_summary_path = results_dir / "evaluation.yml"
    if eval_summary_path.exists():
        try:
            with open(eval_summary_path, 'r') as f:
                results = yaml.safe_load(f)
                
            st.subheader("Summary Results")
            
            # Display metrics
            if 'accuracy' in results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Micro Accuracy", f"{results['accuracy']['micro']:.2%}")
                with col2:
                    st.metric("Macro Accuracy", f"{results['accuracy']['macro']:.2%}")
                with col3:
                    st.metric("P50 Latency", f"{results['p50_latency']:.2f}s")
            
            # Accuracy by use case
            if 'use_case' in results['accuracy']:
                use_case_data = []
                for use_case, acc in results['accuracy']['items']():
                    use_case_data.append({
                        'Use Case': use_case,
                        'Accuracy': f"{acc['macro']:.2%}",
                        'Reject Rate': f"{results.get('reject_rate', {}).get(use_case, 0):.2%}",
                        'P50 Latency': f"{results.get('p50_latency', {}).get(use_case, 0):.2f}s"
                    })
                    
                if use_case_data:
                    st.dataframe(pd.DataFrame(use_case_data), use_container_width=True)
            
            # Full results
            with st.expander("Full Results (YAML)"):
                st.code(yaml.dump(results, default_flow_style=False), language='yaml')
                
        except Exception as e:
            st.error(f"Error reading evaluation results: {str(e)}")
            
    # Look for detailed results
    detailed_results = list(results_dir.glob("*/test_result.yml"))
    
    if detailed_results:
        show_detailed_test_result(results_dir, "test_id_str")
        
def show_detailed_test_result(results_dir: Path, test_id: str):
    """Show detailed results for a specific test"""
    
    # Find the result file for the test_id
    result_files = list(results_dir.glob(f"*/{test_id}/test_result.yml"))
    
    if not result_files:
        st.error(f"No detailed results found for test {test_id}")
        return
        
    result_file = result_files[0]
    
    try:
        with open(result_file, 'r') as f:
            result = yaml.safe_load(f)
            
        with st.expander(f"Detailed Results: {test_id}", expanded=True):
            # Input data
            st.markdown(f"**Question:** {result.get('input_data', {}).get('question', '')}")
            st.code(result.get('input_data', {}).get('ground_truth_output', ''), language='text')
            
            # Predictions
            predictions = result.get('predictions', [])
            for i, pred in enumerate(predictions):
                with st.expander(f"Run {pred.get('run_id', i+1)}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Result:** {pred.get('evaluation', {}).get('llm_label', 'Unknown')}")
                        st.markdown(f"**Latency:** {pred.get('latency', 0):.2f}s")
                    
                    with col2:
                        st.markdown(f"**LLM Judge Output:**")
                        st.markdown(f"**Score:** {pred.get('evaluation', {}).get('llm_judge_output', {}).get('SCORE', 'N/A')}")
                        st.markdown(f"**Reason:** {pred.get('evaluation', {}).get('llm_judge_output', {}).get('REASON', 'N/A')}")
                    
                    # Final response
                    if pred.get('final_response'):
                        st.markdown("**Final Response:**")
                        st.text(pred.get('final_response'))
                        
                    # Combined response (for LLM judge)
                    if pred.get('combined_response'):
                        st.markdown("**Combined Response (for LLM Judge):**")
                        st.text(pred.get('combined_response'))
    except Exception as e:
        st.error(f"Error reading detailed results: {str(e)}")