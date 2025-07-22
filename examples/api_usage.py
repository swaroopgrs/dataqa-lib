"""
DataQA Python API Usage Examples

This module demonstrates various ways to use the DataQA Python API for
programmatic agent creation and management.
"""

import asyncio
from pathlib import Path
from typing import List

import dataqa
from dataqa import (
    DataQAClient,
    DataAgent,
    AgentConfig,
    Document,
    create_agent,
    create_agent_async,
    agent_session,
    quick_query,
    quick_query_async,
)


def example_basic_usage():
    """Example 1: Basic synchronous usage with factory function."""
    print("=== Example 1: Basic Synchronous Usage ===")
    
    # Create agent configuration
    config = {
        "name": "sales-agent",
        "description": "Agent for sales data analysis",
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "${OPENAI_API_KEY}",
            "temperature": 0.1
        },
        "knowledge": {
            "provider": "faiss",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "executor": {
            "provider": "inmemory",
            "database_type": "duckdb",
            "max_execution_time": 30.0
        },
        "workflow": {
            "require_approval": True,
            "enable_visualization": True
        }
    }
    
    # Create agent using factory function
    agent = create_agent("sales-agent", config=config)
    
    try:
        # Query the agent
        response = agent.query("Show me total sales for the last quarter")
        print(f"Agent response: {response}")
        
        # Get agent information
        info = agent.get_agent_info()
        print(f"Agent info: {info}")
        
    finally:
        # Clean up
        agent.shutdown()


async def example_async_usage():
    """Example 2: Asynchronous usage with context manager."""
    print("\n=== Example 2: Asynchronous Usage ===")
    
    config_path = Path("config/example_agent.yaml")
    
    # Use async context manager for automatic cleanup
    async with agent_session("analytics-agent", config_path=config_path) as agent:
        # Process multiple queries
        queries = [
            "What are our top 5 products by revenue?",
            "Show me monthly sales trends",
            "Create a chart of customer segments"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = await agent.query(query)
            print(f"Response: {response}")
            
            # Check if approval is needed
            status = await agent.get_conversation_status("default")
            if status.get("pending_approval"):
                print("Operation requires approval...")
                # In a real application, you'd implement approval logic here
                approval_response = await agent.approve_operation("default", approved=True)
                print(f"Approval response: {approval_response}")


async def example_client_management():
    """Example 3: Managing multiple agents with DataQAClient."""
    print("\n=== Example 3: Client Management ===")
    
    async with DataQAClient() as client:
        # Create multiple agents with different configurations
        agents_config = [
            {
                "name": "sales-agent",
                "description": "Sales data analysis",
                "llm": {"provider": "openai", "model": "gpt-4"}
            },
            {
                "name": "marketing-agent", 
                "description": "Marketing analytics",
                "llm": {"provider": "openai", "model": "gpt-3.5-turbo"}
            },
            {
                "name": "finance-agent",
                "description": "Financial reporting",
                "llm": {"provider": "openai", "model": "gpt-4"}
            }
        ]
        
        # Create all agents
        agents = {}
        for config in agents_config:
            agent = await client.create_agent_async(config["name"], config=config)
            agents[config["name"]] = agent
            print(f"Created agent: {config['name']}")
        
        # List all agents
        print(f"Active agents: {client.list_agents()}")
        
        # Query different agents
        queries = [
            ("sales-agent", "What were our Q4 sales?"),
            ("marketing-agent", "Show me campaign performance"),
            ("finance-agent", "Generate monthly P&L summary")
        ]
        
        for agent_name, query in queries:
            print(f"\nQuerying {agent_name}: {query}")
            response = await client.query_async(agent_name, query)
            print(f"Response: {response}")
        
        # Health check all agents
        print("\n=== Health Check ===")
        for agent_name in client.list_agents():
            health = await client.health_check_async(agent_name)
            print(f"{agent_name} health: {health}")


def example_knowledge_management():
    """Example 4: Knowledge base management."""
    print("\n=== Example 4: Knowledge Management ===")
    
    # Create sample documents
    documents = [
        Document(
            content="Sales data is stored in the sales_transactions table with columns: id, date, product_id, customer_id, amount, quantity",
            metadata={"type": "schema", "table": "sales_transactions"},
            source="database_schema.md"
        ),
        Document(
            content="Customer segments are defined as: Premium (>$10k annual), Standard ($1k-$10k), Basic (<$1k)",
            metadata={"type": "business_rule", "domain": "customer"},
            source="business_rules.md"
        ),
        Document(
            content="Q4 typically shows 40% higher sales due to holiday season. Use seasonal adjustments for forecasting.",
            metadata={"type": "insight", "period": "Q4"},
            source="analyst_notes.md"
        )
    ]
    
    with DataQAClient() as client:
        # Create agent
        agent = client.create_agent("knowledge-agent", config={
            "name": "knowledge-agent",
            "llm": {"provider": "openai", "model": "gpt-4"},
            "knowledge": {"provider": "faiss", "top_k": 3}
        })
        
        # Ingest knowledge
        print("Ingesting knowledge documents...")
        client.ingest_knowledge("knowledge-agent", documents)
        
        # Query with knowledge context
        response = client.query("knowledge-agent", "How should I analyze Q4 sales data?")
        print(f"Knowledge-enhanced response: {response}")


async def example_conversation_management():
    """Example 5: Conversation and state management."""
    print("\n=== Example 5: Conversation Management ===")
    
    async with DataQAClient() as client:
        agent = await client.create_agent_async("conversation-agent", config={
            "name": "conversation-agent",
            "workflow": {"conversation_memory": True}
        })
        
        # Start multiple conversations
        conversations = ["user1", "user2", "user3"]
        
        for conv_id in conversations:
            print(f"\n--- Conversation {conv_id} ---")
            
            # Initial query
            response1 = await client.query_async("conversation-agent", 
                                                "Load sales data for analysis", 
                                                conversation_id=conv_id)
            print(f"Query 1: {response1}")
            
            # Follow-up query (should have context)
            response2 = await client.query_async("conversation-agent",
                                                "Now show me the top products",
                                                conversation_id=conv_id)
            print(f"Query 2: {response2}")
            
            # Get conversation history
            history = await client.get_conversation_history_async("conversation-agent", conv_id)
            print(f"Conversation history length: {len(history)}")
            
            # Get conversation status
            status = await client.get_conversation_status_async("conversation-agent", conv_id)
            print(f"Conversation status: {status}")


def example_quick_operations():
    """Example 6: Quick operations for simple use cases."""
    print("\n=== Example 6: Quick Operations ===")
    
    # Quick synchronous query
    response = quick_query(
        "What is the average order value?",
        config_path="config/example_agent.yaml"
    )
    print(f"Quick query response: {response}")


async def example_quick_operations_async():
    """Example 6b: Async quick operations."""
    # Quick asynchronous query
    response = await quick_query_async(
        "Show me customer distribution by region",
        agent_name="quick-analytics",
        llm={"provider": "openai", "model": "gpt-3.5-turbo"}
    )
    print(f"Quick async response: {response}")


def example_error_handling():
    """Example 7: Error handling and recovery."""
    print("\n=== Example 7: Error Handling ===")
    
    try:
        with DataQAClient() as client:
            # Try to create agent with invalid config
            try:
                agent = client.create_agent("invalid-agent", config={
                    "name": "invalid-agent",
                    "llm": {"provider": "invalid_provider"}
                })
            except Exception as e:
                print(f"Agent creation failed as expected: {e}")
            
            # Create valid agent
            agent = client.create_agent("error-test-agent", config={
                "name": "error-test-agent",
                "llm": {"provider": "openai", "model": "gpt-4"}
            })
            
            # Try to query non-existent agent
            try:
                response = client.query("non-existent", "test query")
            except ValueError as e:
                print(f"Query failed as expected: {e}")
            
            # Valid query
            response = client.query("error-test-agent", "test query")
            print(f"Valid query succeeded: {response}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")


async def example_advanced_configuration():
    """Example 8: Advanced configuration options."""
    print("\n=== Example 8: Advanced Configuration ===")
    
    # Create agent with advanced configuration
    advanced_config = AgentConfig(
        name="advanced-agent",
        description="Advanced agent with custom settings",
        llm={
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 2000,
            "timeout": 60.0,
            "max_retries": 5
        },
        knowledge={
            "provider": "faiss",
            "embedding_model": "all-mpnet-base-v2",
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "top_k": 10,
            "similarity_threshold": 0.8
        },
        executor={
            "provider": "inmemory",
            "max_execution_time": 60.0,
            "max_memory_mb": 1024,
            "max_rows": 50000,
            "allow_file_access": False,
            "allowed_imports": ["pandas", "numpy", "matplotlib", "seaborn", "scipy"]
        },
        workflow={
            "strategy": "react",
            "max_iterations": 15,
            "require_approval": True,
            "auto_approve_safe": False,
            "conversation_memory": True,
            "max_context_length": 8000,
            "enable_visualization": True,
            "debug_mode": True
        },
        log_level="DEBUG",
        data_dir=Path("./custom_data"),
        cache_dir=Path("./custom_cache")
    )
    
    async with DataQAClient() as client:
        agent = await client.create_agent_async("advanced-agent", config=advanced_config)
        
        # Test advanced features
        response = await client.query_async("advanced-agent", 
                                          "Perform complex analysis with visualization")
        print(f"Advanced query response: {response}")
        
        # Check detailed agent info
        info = agent.get_agent_info()
        print(f"Advanced agent info: {info}")


def main():
    """Run all examples."""
    print("DataQA Python API Usage Examples")
    print("=" * 50)
    
    # Synchronous examples
    example_basic_usage()
    example_knowledge_management()
    example_quick_operations()
    example_error_handling()
    
    # Asynchronous examples
    asyncio.run(example_async_usage())
    asyncio.run(example_client_management())
    asyncio.run(example_conversation_management())
    asyncio.run(example_quick_operations_async())
    asyncio.run(example_advanced_configuration())
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()