import streamlit as st
import os
import sys
from pathlib import Path
from typing import Dict, Any
import shutil

def show():
    st.header("Project Manager")
    st.markdown("*Manage your DataQA projects and configurations*")

    # Initialize session state
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    if 'projects_dir' not in st.session_state:
        st.session_state.projects_dir = Path.cwd() / "use_cases"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Projects")

        # Projects directory selection
        projects_dir_input = st.text_input(
            "Projects Directory",
            value=str(st.session_state.projects_dir),
            help="Directory where your use case projects are stored"
        )
        st.session_state.projects_dir = Path(projects_dir_input)

        # Create projects directory if it doesn't exist
        if not st.session_state.projects_dir.exists():
            if st.button("Create Projects Directory"):
                os.makedirs(st.session_state.projects_dir, exist_ok=True)
                st.rerun()

        # List existing projects
        if st.session_state.projects_dir.exists():
            projects = [d for d in st.session_state.projects_dir.iterdir() if d.is_dir()]
            project_names = [p.name for p in projects]

            if projects:
                selected_project_name = st.selectbox("Select Project", project_names)
                if st.button("Load Project"):
                    st.session_state.current_project = st.session_state.projects_dir / selected_project_name
                    st.success(f"Loaded project '{selected_project_name}'")
                    st.rerun()
            else:
                st.info("No projects found in the selected directory")

        # Create new project
        st.subheader("Create New Project")
        new_project_name = st.text_input("Project Name")
        if st.button("Create Project"):
            if new_project_name:
                create_new_project(st.session_state.projects_dir, new_project_name)
                st.rerun()

    with col2:
        if st.session_state.current_project:
            show_project_details(st.session_state.current_project)
        else:
            st.info("Select or create a project to get started")

def create_new_project(projects_dir: Path, project_name: str):
    """Create a new project with the standard structure"""
    project_path = projects_dir / project_name
    try:
        # Create directory structure
        (project_path / "data").mkdir(parents=True, exist_ok=True)

        # Create default configuration files
        create_default_agent_config(project_path / "agent_config.yml")
        create_default_evaluation_config(project_path / "evaluation_config.yml")
        create_default_data_scanner(project_path / "data_scanner_config.yml")
        create_default_rule_inference(project_path / "rule_inference_config.yml")

        # Create default asset files
        create_default_schema(project_path / "data" / "schema.yml")
        create_default_rules(project_path / "data" / "rules.yml")
        create_default_examples(project_path / "data" / "examples.yml")
        create_default_evaluation_data(project_path / "data" / "evaluation_data.yml")

        st.success(f"Created project '{project_name}'")
        st.session_state.current_project = project_path

    except Exception as e:
        st.error(f"Error creating project: {str(e)}")

def show_project_details(project_path: Path):
    """Show details of the current project"""
    st.subheader(f"Project: {project_path.name}")
    
    # Show project structure
    with st.expander("Project Structure", expanded=True):
        show_project_tree(project_path)
        
    # Configuration files
    st.subheader("Configuration Files")
    config_files = {
        "Agent Config": "agent_config.yml",
        "Evaluation Config": "evaluation_config.yml",
        "Data Scanner Config": "data_scanner_config.yml",
        "Rule Inference Config": "rule_inference_config.yml"
    }
    
    for name, filename in config_files.items():
        file_path = project_path / filename
        if file_path.exists():
            with st.expander(name):
                if st.button("Edit", key=f"edit_{filename}"):
                    edit_config_file(file_path)
                    
                # Show file content
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    st.code(content, language='yaml')
                except Exception as e:
                    st.error(f"Error reading {filename}: {str(e)}")

    # Asset files
    st.subheader("Asset Files")
    asset_files = {
        "Schema": "data/schema.yml",
        "Rules": "data/rules.yml",
        "Examples": "data/examples.yml",
        "Evaluation Data": "data/evaluation_data.yml"
    }

    for name, filename in asset_files.items():
        file_path = project_path / filename
        if file_path.exists():
            with st.expander(name):
                if st.button("Edit", key=f"edit_{filename}"):
                    edit_config_file(file_path)
                    
                # Show file content (truncated for large files)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if len(content) > 2000:
                            st.code(content[:2000] + "\n...", language='yaml')
                            st.info(f"File is large ({len(content)} characters). Showing first 2000 characters.")
                        else:
                            st.code(content, language='yaml')
                except Exception as e:
                    st.error(f"Error reading {filename}: {str(e)}")

def show_project_tree(path: Path, prefix: str = ""):
    """Simple file tree display"""
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = prefix + ("└── " if is_last else "├── ")
        
        st.text(f"{current_prefix}{item.name}")
        if item.is_dir() and item.name != "__pycache__":
            next_prefix = prefix + ("    " if is_last else "│   ")
            show_tree(item, next_prefix)

def edit_config_file(file_path: Path):
    """Simple file editor"""
    st.subheader(f"Edit {file_path.name}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        edited_content = st.text_area(
            "File Content",
            value=content,
            height=400,
            key=f"editor_{file_path.name}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes", key=f"save_{file_path.name}"):
                try:
                    with open(file_path, 'w') as f:
                        f.write(edited_content)
                    st.success("File saved successfully!")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")

        with col2:
            if st.button("Validate YAML", key=f"validate_{file_path.name}"):
                try:
                    yaml.safe_load(edited_content)
                    st.success("Valid YAML!")
                except yaml.YAMLError as e:
                    st.error(f"Invalid YAML: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Default configuration templates
def create_default_agent_config(file_path: Path):
    config = {
        'agent_name': 'my_agent',
        'use_case_name': 'My Use Case',
        'use_case_description': 'Describe your use case here.',
        'llm_configs': {
            'gpt-4.1': {
                'type': 'dataqa.core.llm.openai.AzureOpenAI',
                'config': {
                    'model': 'gpt-4.1-2025-04-14',
                    'api_version': '2024-08-01-preview',
                    'api_type': 'azure_ad',
                    'temperature': 0
                }
            }
        },
        'llm': {'default': 'gpt-4.1'},
        'resource_manager_config': {
            'type': 'dataqa.components.resource_manager.resource_manager.ResourceManager',
            'config': {'asset_directory': 'CONFIG_DIR/data'}
        },
        'retriever_config': {
            'type': 'dataqa.core.components.retriever.base_retriever.AllRetriever',
            'config': {
                'name': 'all_retriever',
                'retrieval_method': 'all',
                'module_names': ['planner', 'retrieval_worker', 'analytics_worker', 'plot_worker']
            }
        },
        'workers': {
            'retrieval_worker': {
                'sql_execution_config': {
                    'name': 'sql_executor',
                    'data_files': [
                        {'path': 'CONFIG_DIR/data/my_table.csv', 'table_name': 'my_table'}
                    ]
                }
            }
        }
    }
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def create_default_evaluation_config(file_path: Path):
    config = {
        'use_case_config': {
            'name': 'my_use_case',
            'cwd_agent_config.yml': 'cwd_agent_config.yml'
        },
        'test_data_file': 'evaluation_data.yml',
        'output': 'output/evaluation',
        'log': 'output/evaluation.log',
        'run_llm_eval': True,
        'run_prediction': True,
        'num_run': 1,
        'batch_size': 4,
        'resume': False,
        'debug': False,
        'llm_configs': {
            'gpt-4.1-2025-04-14': {
                # llm config
            }
        },
        'llm': {'default': 'gpt-4.1-2025-04-14'}
    }
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def create_default_data_scanner(file_path: Path):
    config = {
        'executor_config': {
            'name': 'sql_executor',
            'data_files': [
                {'path': 'data/my_table.csv', 'table_name': 'my_table', 'date_columns': []}
            ],
            'backend': 'duckdb'
        },
        'categorical_columns': {'my_table': []},
        'database_type': 'duckdb',
        'output_path': 'data',
        'parameters': {
            'column_example_value_count': 5,
            'metadata_inference_sample_row_count': 50
        }
    }
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
def create_default_rule_inference(file_path: Path):
    config = {
        'config_path': 'agent_config.yml',
        'original_config_file': 'agent_config.yml',
        'test_data_file': 'data/evaluation_data.yml',
        'output_file_path': 'output/rule_inference',
        'max_iteration': 3,
        'multi_tenant_subscription': False
    }
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def create_default_schema(file_path: Path):
    schema = {
        'tables': [
            {
                'table_name': 'my_table',
                'description': 'Description of my table',
                'tags': [],
                'primary_keys': [],
                'foreign_keys': [],
                'columns': [
                    {
                        'column_name': 'id',
                        'description': 'Primary key identifier',
                        'example_values': [1, 2, 3]
                    },
                    {
                        'name': 'name',
                        'type': 'TEXT',
                        'description': 'Name field',
                        'example_value': ['Alice', 'Bob', 'Charlie']
                    }
                ]
            }
        ]
    }
    with open(file_path, 'w') as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

def create_default_rules(file_path: Path):
    rules = {
        'rules': [
            {
                'rule_name': 'general_guidelines',
                'module_name': 'planner',
                'instructions': 'Add your business rules and guidelines here.',
                'tags': [],
                'search_content': ''
            }
        ]
    }
    with open(file_path, 'w') as f:
        yaml.dump(rules, f, default_flow_style=False, sort_keys=False)

def create_default_examples(file_path: Path):
    examples = {
        'examples': [
            {
                'query': 'Show me all records',
                'module_name': 'retrieval_worker',
                'example': {
                    'question': 'Show me all records',
                    'code': 'SELECT * FROM my_table LIMIT 20;',
                    'reasoning': 'Simple query to retrieve all records from the table with a performance limit.'
                },
                'tags': [],
                'search_content': ''
            }
        ]
    }
    with open(file_path, 'w') as f:
        yaml.dump(examples, f, default_flow_style=False, sort_keys=False)

def create_default_evaluation_data(file_path: Path):
    eval_data = {
        'metadata': {
            'use_case': 'my_use_case',
            'as_of_date': '20240101',
            'schema_file': 'my_table.csv'
        },
        'data': [
            {
                'id': 'test_001',
                'active': True,
                'date_created': '',
                'previous_question_id': '',
                'question': 'how many records are in the table?',
                'solution': [
                    {
                        'worker': 'retrieval_worker',
                        'function_name': 'sql_executor',
                        'function_args': {
                            'sql': 'SELECT count(*) as count FROM my_table;'
                        }
                    }
                ],
                'ground_truth_output': '{"count": [n]}\n\n| | count |\n|---|-------|\n| 0 | n |\n',
                'component_groundtruth': {
                    'instruction_for_llm_judge': '',
                    'human_validated': True,
                    'labels': ['sql', 'easy']
                }
            }
        ]
    }
    with open(file_path, 'w') as f:
        yaml.dump(eval_data, f, default_flow_style=False, sort_keys=False)