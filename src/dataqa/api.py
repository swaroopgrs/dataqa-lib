"""
High-level Python API for DataQA framework.

This module provides convenient factory functions, context managers, and async
support for programmatic agent creation and management. It serves as the main
entry point for developers using DataQA in their applications.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .agent.agent import DataAgent, create_agent_from_config
from .config.loader import load_agent_config
from .config.models import AgentConfig
from .models.document import Document
from .models.message import Message

logger = logging.getLogger(__name__)


class DataQAClient:
    """High-level client for DataQA framework.
    
    This class provides a convenient interface for creating and managing
    DataQA agents with automatic resource management and async support.
    
    Example:
        ```python
        # Synchronous usage
        client = DataQAClient()
        agent = client.create_agent("my-agent", config_path="config/agent.yaml")
        response = client.query(agent, "Show me sales data for last month")
        
        # Async usage
        async with DataQAClient() as client:
            agent = await client.create_agent_async("my-agent", config_path="config/agent.yaml")
            response = await client.query_async(agent, "Show me sales data for last month")
        ```
    """
    
    def __init__(self, default_config_dir: Optional[Path] = None):
        """Initialize the DataQA client.
        
        Args:
            default_config_dir: Default directory for configuration files
        """
        self.default_config_dir = default_config_dir or Path("config")
        self._agents: Dict[str, DataAgent] = {}
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def __enter__(self) -> 'DataQAClient':
        """Enter synchronous context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit synchronous context manager."""
        self.shutdown()
    
    async def __aenter__(self) -> 'DataQAClient':
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.shutdown_async()
    
    def create_agent(
        self,
        name: str,
        config: Optional[Union[AgentConfig, Dict[str, Any], str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> DataAgent:
        """Create a DataQA agent synchronously.
        
        Args:
            name: Agent name/identifier
            config: Agent configuration (AgentConfig, dict, or YAML string)
            config_path: Path to configuration file
            **kwargs: Additional configuration parameters
            
        Returns:
            Initialized DataAgent instance
            
        Raises:
            ValueError: If neither config nor config_path is provided
            FileNotFoundError: If config file is not found
        """
        # Run async version in sync context
        if self._event_loop is None:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        return self._event_loop.run_until_complete(
            self.create_agent_async(name, config, config_path, **kwargs)
        )
    
    async def create_agent_async(
        self,
        name: str,
        config: Optional[Union[AgentConfig, Dict[str, Any], str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> DataAgent:
        """Create a DataQA agent asynchronously.
        
        Args:
            name: Agent name/identifier
            config: Agent configuration (AgentConfig, dict, or YAML string)
            config_path: Path to configuration file
            **kwargs: Additional configuration parameters
            
        Returns:
            Initialized DataAgent instance
            
        Raises:
            ValueError: If neither config nor config_path is provided
            FileNotFoundError: If config file is not found
        """
        logger.info(f"Creating agent: {name}")
        
        # Load configuration
        agent_config = await self._load_config(config, config_path, name, **kwargs)
        
        # Create agent
        agent = await create_agent_from_config(agent_config)
        
        # Store agent reference
        self._agents[name] = agent
        
        logger.info(f"Agent created successfully: {name}")
        return agent
    
    def get_agent(self, name: str) -> Optional[DataAgent]:
        """Get an existing agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            DataAgent instance or None if not found
        """
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all created agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def query(
        self,
        agent: Union[DataAgent, str],
        query: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """Query an agent synchronously.
        
        Args:
            agent: DataAgent instance or agent name
            query: User query
            conversation_id: Optional conversation ID
            
        Returns:
            Agent response
            
        Raises:
            ValueError: If agent is not found
        """
        if self._event_loop is None:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        return self._event_loop.run_until_complete(
            self.query_async(agent, query, conversation_id)
        )
    
    async def query_async(
        self,
        agent: Union[DataAgent, str],
        query: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """Query an agent asynchronously.
        
        Args:
            agent: DataAgent instance or agent name
            query: User query
            conversation_id: Optional conversation ID
            
        Returns:
            Agent response
            
        Raises:
            ValueError: If agent is not found
        """
        # Resolve agent
        if isinstance(agent, str):
            agent_instance = self._agents.get(agent)
            if agent_instance is None:
                raise ValueError(f"Agent not found: {agent}")
        else:
            agent_instance = agent
        
        # Process query
        return await agent_instance.query(query, conversation_id)
    
    def approve_operation(
        self,
        agent: Union[DataAgent, str],
        conversation_id: str,
        approved: bool = True,
        reason: Optional[str] = None
    ) -> str:
        """Approve or deny a pending operation synchronously.
        
        Args:
            agent: DataAgent instance or agent name
            conversation_id: Conversation ID
            approved: Whether to approve the operation
            reason: Optional reason for the decision
            
        Returns:
            Response after processing approval
        """
        if self._event_loop is None:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        return self._event_loop.run_until_complete(
            self.approve_operation_async(agent, conversation_id, approved, reason)
        )
    
    async def approve_operation_async(
        self,
        agent: Union[DataAgent, str],
        conversation_id: str,
        approved: bool = True,
        reason: Optional[str] = None
    ) -> str:
        """Approve or deny a pending operation asynchronously.
        
        Args:
            agent: DataAgent instance or agent name
            conversation_id: Conversation ID
            approved: Whether to approve the operation
            reason: Optional reason for the decision
            
        Returns:
            Response after processing approval
        """
        # Resolve agent
        if isinstance(agent, str):
            agent_instance = self._agents.get(agent)
            if agent_instance is None:
                raise ValueError(f"Agent not found: {agent}")
        else:
            agent_instance = agent
        
        # Process approval
        return await agent_instance.approve_operation(conversation_id, approved, reason)
    
    def ingest_knowledge(
        self,
        agent: Union[DataAgent, str],
        documents: List[Document]
    ) -> None:
        """Ingest documents into an agent's knowledge base synchronously.
        
        Args:
            agent: DataAgent instance or agent name
            documents: List of documents to ingest
        """
        if self._event_loop is None:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        self._event_loop.run_until_complete(
            self.ingest_knowledge_async(agent, documents)
        )
    
    async def ingest_knowledge_async(
        self,
        agent: Union[DataAgent, str],
        documents: List[Document]
    ) -> None:
        """Ingest documents into an agent's knowledge base asynchronously.
        
        Args:
            agent: DataAgent instance or agent name
            documents: List of documents to ingest
        """
        # Resolve agent
        if isinstance(agent, str):
            agent_instance = self._agents.get(agent)
            if agent_instance is None:
                raise ValueError(f"Agent not found: {agent}")
        else:
            agent_instance = agent
        
        # Ingest documents
        await agent_instance.ingest_knowledge(documents)
    
    def get_conversation_history(
        self,
        agent: Union[DataAgent, str],
        conversation_id: str
    ) -> List[Message]:
        """Get conversation history synchronously.
        
        Args:
            agent: DataAgent instance or agent name
            conversation_id: Conversation ID
            
        Returns:
            List of messages in the conversation
        """
        if self._event_loop is None:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        return self._event_loop.run_until_complete(
            self.get_conversation_history_async(agent, conversation_id)
        )
    
    async def get_conversation_history_async(
        self,
        agent: Union[DataAgent, str],
        conversation_id: str
    ) -> List[Message]:
        """Get conversation history asynchronously.
        
        Args:
            agent: DataAgent instance or agent name
            conversation_id: Conversation ID
            
        Returns:
            List of messages in the conversation
        """
        # Resolve agent
        if isinstance(agent, str):
            agent_instance = self._agents.get(agent)
            if agent_instance is None:
                raise ValueError(f"Agent not found: {agent}")
        else:
            agent_instance = agent
        
        # Get conversation history
        return await agent_instance.get_conversation_history(conversation_id)
    
    def health_check(self, agent: Union[DataAgent, str]) -> Dict[str, Any]:
        """Perform health check on an agent synchronously.
        
        Args:
            agent: DataAgent instance or agent name
            
        Returns:
            Health status information
        """
        if self._event_loop is None:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        return self._event_loop.run_until_complete(
            self.health_check_async(agent)
        )
    
    async def health_check_async(self, agent: Union[DataAgent, str]) -> Dict[str, Any]:
        """Perform health check on an agent asynchronously.
        
        Args:
            agent: DataAgent instance or agent name
            
        Returns:
            Health status information
        """
        # Resolve agent
        if isinstance(agent, str):
            agent_instance = self._agents.get(agent)
            if agent_instance is None:
                raise ValueError(f"Agent not found: {agent}")
        else:
            agent_instance = agent
        
        # Perform health check
        return await agent_instance.health_check()
    
    def shutdown(self) -> None:
        """Shutdown all agents and clean up resources synchronously."""
        if self._event_loop is None:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        self._event_loop.run_until_complete(self.shutdown_async())
        
        # Close event loop
        if self._event_loop and not self._event_loop.is_closed():
            self._event_loop.close()
            self._event_loop = None
    
    async def shutdown_async(self) -> None:
        """Shutdown all agents and clean up resources asynchronously."""
        logger.info("Shutting down DataQA client...")
        
        # Shutdown all agents
        for name, agent in self._agents.items():
            try:
                await agent.shutdown()
                logger.info(f"Agent shutdown complete: {name}")
            except Exception as e:
                logger.error(f"Error shutting down agent {name}: {e}")
        
        # Clear agent references
        self._agents.clear()
        
        logger.info("DataQA client shutdown complete")
    
    async def _load_config(
        self,
        config: Optional[Union[AgentConfig, Dict[str, Any], str, Path]],
        config_path: Optional[Union[str, Path]],
        name: str,
        **kwargs
    ) -> AgentConfig:
        """Load and validate agent configuration.
        
        Args:
            config: Configuration object, dict, or YAML string
            config_path: Path to configuration file
            name: Agent name
            **kwargs: Additional configuration parameters
            
        Returns:
            Validated AgentConfig instance
            
        Raises:
            ValueError: If neither config nor config_path is provided
            FileNotFoundError: If config file is not found
        """
        if config is not None:
            # Handle different config types
            if isinstance(config, AgentConfig):
                agent_config = config
            elif isinstance(config, dict):
                agent_config = AgentConfig(**config)
            elif isinstance(config, (str, Path)):
                # Treat as YAML content or file path
                if Path(config).exists():
                    agent_config = load_agent_config(Path(config))
                else:
                    # Treat as YAML string
                    import yaml
                    config_dict = yaml.safe_load(config)
                    agent_config = AgentConfig(**config_dict)
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
        elif config_path is not None:
            # Load from file
            config_file = Path(config_path)
            if not config_file.is_absolute():
                config_file = self.default_config_dir / config_file
            
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
            agent_config = load_agent_config(config_file)
        else:
            raise ValueError("Either config or config_path must be provided")
        
        # Override with kwargs
        if kwargs:
            config_dict = agent_config.model_dump()
            config_dict.update(kwargs)
            agent_config = AgentConfig(**config_dict)
        
        # Ensure name is set
        if not agent_config.name:
            agent_config.name = name
        
        return agent_config


# Convenience factory functions

def create_agent(
    name: str,
    config: Optional[Union[AgentConfig, Dict[str, Any], str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> DataAgent:
    """Create a DataQA agent with default client.
    
    This is a convenience function for simple use cases where you don't need
    to manage multiple agents or use async operations.
    
    Args:
        name: Agent name/identifier
        config: Agent configuration
        config_path: Path to configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized DataAgent instance
        
    Example:
        ```python
        agent = create_agent("my-agent", config_path="config/agent.yaml")
        response = agent.query("Show me sales data")
        ```
    """
    client = DataQAClient()
    return client.create_agent(name, config, config_path, **kwargs)


async def create_agent_async(
    name: str,
    config: Optional[Union[AgentConfig, Dict[str, Any], str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> DataAgent:
    """Create a DataQA agent asynchronously with default client.
    
    Args:
        name: Agent name/identifier
        config: Agent configuration
        config_path: Path to configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized DataAgent instance
        
    Example:
        ```python
        agent = await create_agent_async("my-agent", config_path="config/agent.yaml")
        response = await agent.query("Show me sales data")
        ```
    """
    client = DataQAClient()
    return await client.create_agent_async(name, config, config_path, **kwargs)


@asynccontextmanager
async def agent_session(
    name: str,
    config: Optional[Union[AgentConfig, Dict[str, Any], str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> AsyncGenerator[DataAgent, None]:
    """Create an agent within an async context manager for automatic cleanup.
    
    Args:
        name: Agent name/identifier
        config: Agent configuration
        config_path: Path to configuration file
        **kwargs: Additional configuration parameters
        
    Yields:
        Initialized DataAgent instance
        
    Example:
        ```python
        async with agent_session("my-agent", config_path="config/agent.yaml") as agent:
            response = await agent.query("Show me sales data")
            # Agent is automatically cleaned up when exiting the context
        ```
    """
    async with DataQAClient() as client:
        agent = await client.create_agent_async(name, config, config_path, **kwargs)
        try:
            yield agent
        finally:
            await agent.shutdown()


# Convenience functions for common operations

def quick_query(
    query: str,
    config_path: Optional[Union[str, Path]] = None,
    agent_name: str = "quick-agent",
    **config_kwargs
) -> str:
    """Perform a quick query with minimal setup.
    
    This function creates a temporary agent, processes the query, and cleans up.
    Useful for one-off queries or testing.
    
    Args:
        query: User query to process
        config_path: Optional path to configuration file
        agent_name: Name for the temporary agent
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Query response
        
    Example:
        ```python
        response = quick_query(
            "Show me sales data for last month",
            config_path="config/agent.yaml"
        )
        ```
    """
    with DataQAClient() as client:
        # Create config from kwargs if provided, otherwise use config_path
        config = config_kwargs if config_kwargs else None
        agent = client.create_agent(agent_name, config=config, config_path=config_path)
        return client.query(agent, query)


async def quick_query_async(
    query: str,
    config_path: Optional[Union[str, Path]] = None,
    agent_name: str = "quick-agent",
    **config_kwargs
) -> str:
    """Perform a quick query asynchronously with minimal setup.
    
    Args:
        query: User query to process
        config_path: Optional path to configuration file
        agent_name: Name for the temporary agent
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Query response
        
    Example:
        ```python
        response = await quick_query_async(
            "Show me sales data for last month",
            config_path="config/agent.yaml"
        )
        ```
    """
    # Create config from kwargs if provided, otherwise use config_path
    config = config_kwargs if config_kwargs else None
    async with agent_session(agent_name, config=config, config_path=config_path) as agent:
        return await agent.query(query)


# Export main classes and functions
__all__ = [
    "DataQAClient",
    "create_agent",
    "create_agent_async", 
    "agent_session",
    "quick_query",
    "quick_query_async"
]