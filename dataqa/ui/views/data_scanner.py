import streamlit as st
import asyncio
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataqa.core.components.knowledge_extraction.data_scanner import DataScanner

def show():
    st.header("🔍 Data Scanner")
    st.markdown("*Automatically extract schema and metadata from your data sources*")

    # Check if project is loaded
    if 'current_project' not in st.session_state or st.session_state.current_project is None:
        st.warning("Please select a project in the Project Manager first.")
        return

    project_path = st.session_state.current_project
    scanner_config_path = project_path / "data_scanner_config.yml"

    # Configuration section
    st.subheader("Data Scanner Configuration")

    if scanner_config_path.exists():
        try:
            with open(scanner_config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Display and edit configuration
            with st.expander("Configuration", expanded=True):
                # Database type
                config['database_type'] = st.selectbox(
                    "Database Type",
                    ['sqlite', 'duckdb', 'snowflake', 'sqlserver'],
                    index=['sqlite', 'duckdb', 'snowflake', 'sqlserver'].index(
                        config.get('database_type', 'duckdb'))
                )
                
                # Output path
                config['output_path'] = st.text_input(
                    "Output Path",
                    config.get('output_path', str(project_path / "data"))
                )

                # Data files configuration
                st.subheader("Data Files")
                data_files = config.get('executor_config', {}).get('data_files', [])

                # Add new data file
                with st.expander("Add Data File"):
                    new_file_path = st.text_input("File Path")
                    new_table_name = st.text_input("Table Name")
                    if st.button("Add"):
                        if new_file_path and new_table_name:
                            data_files.append({
                                'path': new_file_path,
                                'table_name': new_table_name,
                                'date_columns': []
                            })
                            config['executor_config']['data_files'] = data_files
                            st.success(f"Added {new_table_name}")
                            st.rerun()

                # Display existing data files
                st.subheader("Current Data Files")
                for i, file_config in enumerate(data_files):
                    with st.expander(f"{file_config.get('table_name', f'File {i+1}')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.text(f"Path: {file_config.get('path', '')}")
                        with col2:
                            st.text(f"Table Name: {file_config.get('table_name', '')}")
                        
                        # Check if file exists
                        file_path = file_config.get('path', '')
                        if not Path(file_path).is_absolute():
                            file_path = project_path / file_path
                        
                        if file_path.exists():
                            st.success("✓ File exists")

                            # Show file preview
                            if st.button("Preview Data", key=f"preview_{i}"):
                                show_data_preview(file_path)
                        else:
                            st.error("✗ File not found")
                        
                        with col2:
                            if st.button("Remove", key=f"remove_{i}"):
                                data_files.pop(i)
                                config['executor_config']['data_files'] = data_files
                                st.rerun()
                
                # Categorical columns configuration
                st.subheader("Categorical Columns")
                categorical_cols = config.get('categorical_columns', {})
                for table_name in [f.get('table_name', '') for f in data_files]:
                    if table_name:
                        cols_text_input = st.text_input(
                            f"Categorical columns for {table_name} (comma-separated)",
                            value=",".join(categorical_cols.get(table_name, []))
                        )
                        key = f"cat_cols_{table_name}"

                        if cols:
                            categorical_cols[table_name] = [col.strip() for col in cols.split(',')]
                        else:
                            categorical_cols[table_name] = []
                
                config['categorical_columns'] = categorical_cols
                
                # Parameters
                st.subheader("Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    config.setdefault('parameters', {})
                    config['parameters']['column_example_value_count'] = st.number_input(
                        "Sample Values Count",
                        min_value=1, max_value=20,
                        value=config.get('column_example_value_count', 5)
                    )
                with col2:
                    config['parameters']['metadata_inference_sample_row_count'] = st.number_input(
                        "Sample Rows for Inference",
                        min_value=10, max_value=1000,
                        value=config.get('metadata_inference_sample_row_count', 50)
                    )
                
                # Save configuration
                if st.button("💾 Save Configuration"):
                    with open(scanner_config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    st.success("Configuration saved!")
                
                # Show full config
                with st.expander("Full Configuration (YAML)"):
                    st.code(yaml.dump(config, default_flow_style=False), language='yaml')
        except Exception as e:
            st.error(f"Error reading scanner config: {str(e)}")
            return
    else:
        st.error(f"Data scanner configuration not found: {scanner_config_path}")
        return

    # Environment setup for LLM Inference
    st.subheader("Environment Setup for LLM Inference")
    st.info("LLM credentials are required for metadata inference (generating descriptions)")
    
    col1, col2 = st.columns(2)
    with col1:
        cert_path = st.text_input("Certificate Path (optional)",
                                value=os.environ.get("CERT_PATH", ""))
        if cert_path:
            os.environ["CERT_PATH"] = cert_path
    with col2:
        api_base = st.text_input("OpenAI API Base",
                               value=os.environ.get("OPENAI_API_BASE", ""))
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base
            
    # Run data scanner
    st.subheader("Run Data Scanner")
    col1, col2 = st.columns(2)
    with col1:
        extract_only = st.checkbox("Extract Schema Only", value=False,
                                 help="Skip LLM-based metadata inference")
    with col2:
        overwrite = st.checkbox("Overwrite Existing Files", value=False)
    
    if st.button("🏃‍♂️ Run Data Scanner"):
        run_data_scanner_pipeline(config, extract_only, overwrite)

    # Results section
    st.subheader("Generated Schema Files")
    output_path = Path(config.get('output_path', project_path / "data"))

    # Show generated files
    schema_files = {
        "Inferred Schema": output_path / "schema_inferred.yml",
        "Final Schema": output_path / "schema.yml"
    }

    for name, file_path in schema_files.items():
        if file_path.exists():
            with st.expander(f"{name}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info(f"File: {file_path} (Size: {file_path.stat().st_size} bytes, Modified: {pd.to_datetime(file_path.stat().st_mtime, unit='s')})")
                
                with col2:
                    if st.button("Use as Final Schema", key=f"use_{name}"):
                        copy_to_final_schema(file_path, project_path / "data" / "schema.yml")
                
                # Show content
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if len(content) > 5000:
                        st.info(f"File is large ({len(content)} characters). Showing first 5000 characters.")
                        st.code(content[:5000] + "\n...", language='yaml')
                    else:
                        st.code(content, language='yaml')

                    if st.button("Download Full File", key=f"download_{name}"):
                        download_button = st.download_button(
                            label="Download",
                            data=content,
                            file_name=file_path.name,
                            mime='text/yaml'
                        )
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

def show_data_preview(file_path: Path):
    """Show preview of a data file"""
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=100)
            st.subheader(f"Data Preview ({file_path.name})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows (sample)", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                try:
                    full_df = pd.read_csv(file_path)
                    st.metric("Total Rows", len(full_df))
                except:
                    st.metric("Total Rows", "Unknown")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column Information
            with st.expander("Column Information"):
                col_info = []
                for col in df.columns:
                    col_info.append({
                        'Column': col,
                        'Type': str(df[col].dtype),
                        'Non-Null': df[col].count(),
                        'Unique': df[col].nunique(),
                        'Sample Values': ', '.join([str(x) for x in df[col].dropna().unique()[:5]])
                    })
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        else:
            st.warning(f"Preview not supported for {file_path.suffix} files")
    except Exception as e:
        st.error(f"Error previewing file: {str(e)}")

def run_data_scanner_pipeline(config: dict, extract_only: bool, overwrite: bool):
    """Run the data scanner pipeline"""
    with st.spinner("Running data scanner..."):
        try:
            # Check output paths
            output_path = Path(config.get('output_path', ''))
            extracted_file = output_path / "schema_extracted.yml"
            inferred_file = output_path / "schema_inferred.yml"
            
            if not overwrite:
                if extracted_file.exists() or inferred_file.exists():
                    st.warning("Output files already exist. Enable 'Overwrite Existing Files' to proceed.")
                    return
            
            # Create data scanner
            scanner = DataScanner(config)
            
            # Extract schema
            st.info("Extracting schema from data sources...")
            asyncio.run(scanner.extract_schema())
            
            st.success("✅ Schema extraction completed!")
            
            if not extract_only:
                # Infer metadata using LLM
                st.info("Inferring metadata using LLM...")
                
                if not os.environ.get("OPENAI_API_BASE"):
                    st.error("OpenAI API Base is required for metadata inference")
                    return
                try:
                    asyncio.run(scanner.infer_metadata())
                    st.success("✅ Metadata inference completed")
                except Exception as e:
                    st.error(f"Metadata inference failed: {str(e)}")
                    st.info("Schema extraction was successful. You can use the extracted schema without metadata inference.")
            
            st.success("Data scanner completed successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"Error running data scanner: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def copy_to_final_schema(source_path: Path, target_path: Path):
    """Copy schema file to final location"""
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, target_path)
        st.success(f"Copied {source_path.name} to {target_path}")
    except Exception as e:
        st.error(f"Error copying file: {str(e)}")