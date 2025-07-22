"""Main CLI entry point for DataQA."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from ..agent.agent import DataAgent, create_agent_from_config
from ..config.loader import (
    ConfigurationError,
    create_example_config,
    load_agent_config,
    validate_environment,
)
from ..models.document import Document

# Initialize Rich console for beautiful output
console = Console()

# Create the main Typer app
app = typer.Typer(
    name="dataqa",
    help="DataQA - A composable data agent framework for natural language data interaction",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def setup_logging(log_level: str = "INFO", debug: bool = False) -> None:
    """Set up logging with Rich handler."""
    level = logging.DEBUG if debug else getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@app.command()
def run(
    config_path: Path = typer.Option(
        Path("config/example_agent.yaml"),
        "--config", "-c",
        help="Path to agent configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    conversation_id: Optional[str] = typer.Option(
        None,
        "--conversation", "-conv",
        help="Conversation ID for session persistence"
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Enable debug logging"
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve", "-y",
        help="Auto-approve all operations (use with caution)"
    )
) -> None:
    """Run an interactive DataQA agent session."""
    
    setup_logging(debug=debug)
    
    try:
        # Load configuration
        console.print(f"[blue]Loading configuration from {config_path}[/blue]")
        config = load_agent_config(config_path)
        
        # Override approval setting if auto-approve is enabled
        if auto_approve:
            config.workflow.require_approval = False
            console.print("[yellow]Warning: Auto-approval enabled. All operations will execute without confirmation.[/yellow]")
        
        # Run the interactive session
        asyncio.run(_run_interactive_session(config, conversation_id))
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


async def _run_interactive_session(config, conversation_id: Optional[str]) -> None:
    """Run the interactive agent session."""
    
    # Create agent
    console.print("[blue]Initializing DataQA agent...[/blue]")
    agent = await create_agent_from_config(config)
    
    # Display agent info
    info = agent.get_agent_info()
    
    info_table = Table(title="Agent Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Name", info["name"])
    info_table.add_row("Description", info.get("description", "N/A"))
    info_table.add_row("LLM Provider", f"{info['llm_provider']} ({info['llm_model']})")
    info_table.add_row("Knowledge Provider", info["knowledge_provider"])
    info_table.add_row("Executor Provider", info["executor_provider"])
    
    console.print(info_table)
    console.print()
    
    # Health check
    console.print("[blue]Performing health check...[/blue]")
    health = await agent.health_check()
    
    health_table = Table(title="Component Health")
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    
    for component, status in health.items():
        if component == "timestamp":
            continue
        
        if status == "healthy":
            status_text = "[green]✓ Healthy[/green]"
        elif isinstance(status, str) and status.startswith("unhealthy"):
            status_text = f"[red]✗ {status}[/red]"
        else:
            status_text = f"[yellow]? {status}[/yellow]"
        
        health_table.add_row(component.title(), status_text)
    
    console.print(health_table)
    console.print()
    
    # Interactive loop
    console.print("[green]DataQA agent is ready! Type your questions or 'quit' to exit.[/green]")
    console.print("[dim]Commands: /help, /status, /history, /clear, /quit[/dim]")
    console.print()
    
    try:
        while True:
            # Get user input
            query = Prompt.ask("[bold blue]You[/bold blue]")
            
            if not query.strip():
                continue
            
            # Handle special commands
            if query.lower() in ["/quit", "/exit", "quit", "exit"]:
                break
            elif query.lower() == "/help":
                _show_help()
                continue
            elif query.lower() == "/status":
                await _show_status(agent, conversation_id)
                continue
            elif query.lower() == "/history":
                await _show_history(agent, conversation_id)
                continue
            elif query.lower() == "/clear":
                if conversation_id:
                    cleared = await agent.clear_conversation(conversation_id)
                    if cleared:
                        console.print("[green]Conversation history cleared.[/green]")
                    else:
                        console.print("[yellow]No conversation history to clear.[/yellow]")
                else:
                    console.print("[yellow]No conversation ID specified.[/yellow]")
                continue
            
            # Process the query
            console.print("[blue]Agent[/blue]: Processing your query...")
            
            try:
                response = await agent.query(query, conversation_id)
                
                # Display response in a panel
                console.print(Panel(
                    response,
                    title="[bold green]Agent Response[/bold green]",
                    border_style="green"
                ))
                
                # Check if there's a pending approval
                if conversation_id:
                    status = await agent.get_conversation_status(conversation_id)
                    if status.get("pending_approval"):
                        await _handle_approval(agent, conversation_id)
                
            except Exception as e:
                console.print(f"[red]Error processing query: {e}[/red]")
                logging.exception("Query processing error")
    
    finally:
        # Cleanup
        console.print("[blue]Shutting down agent...[/blue]")
        await agent.shutdown()
        console.print("[green]Goodbye![/green]")


def _show_help() -> None:
    """Show help information."""
    help_text = """
[bold]Available Commands:[/bold]

[cyan]/help[/cyan]     - Show this help message
[cyan]/status[/cyan]   - Show conversation status
[cyan]/history[/cyan]  - Show conversation history
[cyan]/clear[/cyan]    - Clear conversation history
[cyan]/quit[/cyan]     - Exit the session

[bold]Usage Tips:[/bold]

• Ask questions about your data in natural language
• The agent will generate and execute SQL/Python code
• Review generated code before approving execution
• Use conversation ID to maintain context across sessions
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))


async def _show_status(agent: DataAgent, conversation_id: Optional[str]) -> None:
    """Show conversation status."""
    if not conversation_id:
        console.print("[yellow]No conversation ID specified.[/yellow]")
        return
    
    status = await agent.get_conversation_status(conversation_id)
    
    if not status.get("exists"):
        console.print("[yellow]No active conversation found.[/yellow]")
        return
    
    status_table = Table(title="Conversation Status")
    status_table.add_column("Property", style="cyan")
    status_table.add_column("Value", style="green")
    
    status_table.add_row("Current Step", status.get("current_step", "N/A"))
    status_table.add_row("Workflow Complete", str(status.get("workflow_complete", False)))
    status_table.add_row("Error Occurred", str(status.get("error_occurred", False)))
    status_table.add_row("Pending Approval", str(status.get("pending_approval", False)))
    status_table.add_row("Iteration Count", str(status.get("iteration_count", 0)))
    status_table.add_row("Message Count", str(status.get("message_count", 0)))
    
    if status.get("error_message"):
        status_table.add_row("Error Message", status["error_message"])
    
    console.print(status_table)


async def _show_history(agent: DataAgent, conversation_id: Optional[str]) -> None:
    """Show conversation history."""
    if not conversation_id:
        console.print("[yellow]No conversation ID specified.[/yellow]")
        return
    
    history = await agent.get_conversation_history(conversation_id)
    
    if not history:
        console.print("[yellow]No conversation history found.[/yellow]")
        return
    
    console.print(f"[bold]Conversation History ({len(history)} messages):[/bold]")
    console.print()
    
    for i, message in enumerate(history[-10:], 1):  # Show last 10 messages
        role_color = {
            "user": "blue",
            "assistant": "green",
            "system": "yellow"
        }.get(message.role, "white")
        
        console.print(f"[{role_color}]{message.role.title()}[/{role_color}]: {message.content[:200]}...")
        if i < len(history[-10:]):
            console.print()


async def _handle_approval(agent: DataAgent, conversation_id: str) -> None:
    """Handle pending approval requests."""
    console.print("[yellow]⚠️  Operation requires approval[/yellow]")
    
    approved = Confirm.ask("Do you want to proceed with the operation?")
    
    try:
        response = await agent.approve_operation(conversation_id, approved)
        console.print(Panel(
            response,
            title="[bold green]Operation Result[/bold green]" if approved else "[bold red]Operation Cancelled[/bold red]",
            border_style="green" if approved else "red"
        ))
    except Exception as e:
        console.print(f"[red]Error processing approval: {e}[/red]")


@app.command()
def ingest(
    config_path: Path = typer.Option(
        Path("config/example_agent.yaml"),
        "--config", "-c",
        help="Path to agent configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    documents: List[Path] = typer.Argument(
        ...,
        help="Paths to documents to ingest into knowledge base"
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive", "-r",
        help="Recursively process directories"
    ),
    file_pattern: str = typer.Option(
        "*.txt,*.md,*.pdf,*.docx",
        "--pattern", "-p",
        help="File pattern for recursive processing (comma-separated)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Enable debug logging"
    )
) -> None:
    """Ingest documents into the agent's knowledge base."""
    
    setup_logging(debug=debug)
    
    try:
        # Load configuration
        console.print(f"[blue]Loading configuration from {config_path}[/blue]")
        config = load_agent_config(config_path)
        
        # Run the ingestion
        asyncio.run(_run_ingestion(config, documents, recursive, file_pattern))
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Ingestion failed: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


async def _run_ingestion(config, documents: List[Path], recursive: bool, file_pattern: str) -> None:
    """Run the document ingestion process."""
    
    # Create agent
    console.print("[blue]Initializing DataQA agent...[/blue]")
    agent = await create_agent_from_config(config)
    
    try:
        # Collect files to process
        files_to_process = []
        
        for doc_path in documents:
            if doc_path.is_file():
                files_to_process.append(doc_path)
            elif doc_path.is_dir() and recursive:
                # Use glob pattern to find files
                # Handle comma-separated patterns like "*.txt,*.md,*.pdf"
                patterns = [p.strip() for p in file_pattern.split(',')]
                for pattern in patterns:
                    files_to_process.extend(doc_path.rglob(pattern))
            else:
                console.print(f"[yellow]Skipping {doc_path} (not a file or recursive not enabled)[/yellow]")
        
        if not files_to_process:
            console.print("[yellow]No files found to process.[/yellow]")
            return
        
        console.print(f"[green]Found {len(files_to_process)} files to process[/green]")
        
        # Process files
        documents_to_ingest = []
        
        for file_path in files_to_process:
            try:
                console.print(f"[blue]Processing {file_path}[/blue]")
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Create document
                doc = Document(
                    content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": file_path.suffix,
                        "size": len(content)
                    },
                    source=str(file_path)
                )
                
                documents_to_ingest.append(doc)
                
            except Exception as e:
                console.print(f"[red]Error processing {file_path}: {e}[/red]")
        
        if not documents_to_ingest:
            console.print("[yellow]No documents could be processed.[/yellow]")
            return
        
        # Ingest documents
        console.print(f"[blue]Ingesting {len(documents_to_ingest)} documents...[/blue]")
        await agent.ingest_knowledge(documents_to_ingest)
        
        console.print(f"[green]Successfully ingested {len(documents_to_ingest)} documents![/green]")
        
    finally:
        await agent.shutdown()


@app.command()
def benchmark(
    config_path: Path = typer.Option(
        Path("config/example_agent.yaml"),
        "--config", "-c",
        help="Path to agent configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    benchmark_file: Optional[Path] = typer.Option(
        None,
        "--benchmark", "-b",
        help="Path to benchmark questions file (JSON/YAML)"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save benchmark results"
    ),
    iterations: int = typer.Option(
        1,
        "--iterations", "-i",
        help="Number of iterations to run each benchmark"
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Enable debug logging"
    )
) -> None:
    """Run benchmarks against the DataQA agent."""
    
    setup_logging(debug=debug)
    
    try:
        # Load configuration
        console.print(f"[blue]Loading configuration from {config_path}[/blue]")
        config = load_agent_config(config_path)
        
        # Run benchmarks
        asyncio.run(_run_benchmarks(config, benchmark_file, output_file, iterations))
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


async def _run_benchmarks(config, benchmark_file: Optional[Path], output_file: Optional[Path], iterations: int) -> None:
    """Run the benchmark process."""
    
    # Create agent
    console.print("[blue]Initializing DataQA agent for benchmarking...[/blue]")
    agent = await create_agent_from_config(config)
    
    try:
        # Load benchmark questions
        if benchmark_file and benchmark_file.exists():
            import json
            import yaml
            
            with open(benchmark_file, 'r') as f:
                if benchmark_file.suffix.lower() == '.json':
                    benchmark_data = json.load(f)
                else:
                    benchmark_data = yaml.safe_load(f)
            
            questions = benchmark_data.get('questions', [])
        else:
            # Use default benchmark questions
            questions = [
                "What tables are available in the database?",
                "Show me the first 5 rows of data",
                "What are the column names and types?",
                "Generate a simple visualization of the data",
                "Calculate basic statistics for numeric columns"
            ]
        
        if not questions:
            console.print("[yellow]No benchmark questions found.[/yellow]")
            return
        
        console.print(f"[green]Running {len(questions)} benchmark questions with {iterations} iterations each[/green]")
        
        results = []
        
        for i, question in enumerate(questions, 1):
            console.print(f"[blue]Benchmark {i}/{len(questions)}: {question}[/blue]")
            
            question_results = []
            
            for iteration in range(iterations):
                console.print(f"[dim]  Iteration {iteration + 1}/{iterations}[/dim]")
                
                import time
                start_time = time.time()
                
                try:
                    response = await agent.query(question, f"benchmark_{i}_{iteration}")
                    end_time = time.time()
                    
                    result = {
                        "question": question,
                        "iteration": iteration + 1,
                        "success": True,
                        "response": response,
                        "execution_time": end_time - start_time,
                        "error": None
                    }
                    
                except Exception as e:
                    end_time = time.time()
                    result = {
                        "question": question,
                        "iteration": iteration + 1,
                        "success": False,
                        "response": None,
                        "execution_time": end_time - start_time,
                        "error": str(e)
                    }
                
                question_results.append(result)
            
            results.extend(question_results)
            
            # Show summary for this question
            successful = sum(1 for r in question_results if r["success"])
            avg_time = sum(r["execution_time"] for r in question_results) / len(question_results)
            
            console.print(f"[green]  Success rate: {successful}/{iterations} ({successful/iterations*100:.1f}%)[/green]")
            console.print(f"[green]  Average time: {avg_time:.2f}s[/green]")
            console.print()
        
        # Overall summary
        total_successful = sum(1 for r in results if r["success"])
        total_tests = len(results)
        overall_avg_time = sum(r["execution_time"] for r in results) / len(results)
        
        summary_table = Table(title="Benchmark Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tests", str(total_tests))
        summary_table.add_row("Successful", str(total_successful))
        summary_table.add_row("Success Rate", f"{total_successful/total_tests*100:.1f}%")
        summary_table.add_row("Average Time", f"{overall_avg_time:.2f}s")
        
        console.print(summary_table)
        
        # Save results if output file specified
        if output_file:
            import json
            from datetime import datetime
            
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "config": config.model_dump(mode='json'),
                "summary": {
                    "total_tests": total_tests,
                    "successful": total_successful,
                    "success_rate": total_successful/total_tests,
                    "average_time": overall_avg_time
                },
                "results": results
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            console.print(f"[green]Results saved to {output_file}[/green]")
    
    finally:
        await agent.shutdown()


@app.command()
def config(
    action: str = typer.Argument(
        ...,
        help="Action to perform: create, validate, show-env"
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file"
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for created configuration"
    )
) -> None:
    """Manage DataQA agent configurations."""
    
    if action == "create":
        _create_config(output_path or Path("config/new_agent.yaml"))
    elif action == "validate":
        if not config_path:
            console.print("[red]Configuration path required for validation[/red]")
            raise typer.Exit(1)
        _validate_config(config_path)
    elif action == "show-env":
        _show_environment()
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: create, validate, show-env")
        raise typer.Exit(1)


def _create_config(output_path: Path) -> None:
    """Create a new example configuration."""
    try:
        config = create_example_config(output_path)
        console.print(f"[green]Created example configuration at {output_path}[/green]")
        console.print(f"[blue]Agent name: {config.name}[/blue]")
        console.print("[yellow]Remember to set your API keys in environment variables![/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to create configuration: {e}[/red]")
        raise typer.Exit(1)


def _validate_config(config_path: Path) -> None:
    """Validate a configuration file."""
    try:
        config = load_agent_config(config_path)
        console.print(f"[green]✓ Configuration is valid[/green]")
        console.print(f"[blue]Agent: {config.name}[/blue]")
        console.print(f"[blue]LLM: {config.llm.provider.value} ({config.llm.model})[/blue]")
        console.print(f"[blue]Knowledge: {config.knowledge.provider.value}[/blue]")
        console.print(f"[blue]Executor: {config.executor.provider.value}[/blue]")
    except ConfigurationError as e:
        console.print(f"[red]✗ Configuration is invalid: {e}[/red]")
        raise typer.Exit(1)


def _show_environment() -> None:
    """Show environment variable status."""
    env_status = validate_environment()
    
    env_table = Table(title="Environment Variables")
    env_table.add_column("Variable", style="cyan")
    env_table.add_column("Status", style="green")
    
    for var_name, is_set in env_status.items():
        status = "[green]✓ Set[/green]" if is_set else "[red]✗ Not Set[/red]"
        env_table.add_row(var_name, status)
    
    console.print(env_table)


@app.command()
def version() -> None:
    """Show DataQA version information."""
    try:
        import dataqa
        version = getattr(dataqa, '__version__', '0.1.0')
    except ImportError:
        version = '0.1.0'
    
    console.print(f"[bold green]DataQA version {version}[/bold green]")
    console.print("A composable data agent framework for natural language data interaction")


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()