"""Main DataAgent class for high-level agent orchestration."""

import logging
from typing import Any, Dict, List, Optional

from ..config.models import AgentConfig
from ..primitives.executor import ExecutorPrimitive
from ..primitives.faiss_knowledge import FAISSKnowledge
from ..primitives.in_memory_executor import InMemoryExecutor
from ..primitives.knowledge import KnowledgePrimitive
from ..primitives.llm import LLMInterface, create_llm_interface
from ..models.document import Document
from ..models.message import Message
from .state import SharedState
from .workflow import DataAgentWorkflow

logger = logging.getLogger(__name__)


class DataAgent:
    """Main DataAgent class that orchestrates all components.
    
    This class provides a high-level interface for creating and managing
    data agents. It handles component initialization, workflow orchestration,
    and conversation state management.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[LLMInterface] = None,
        knowledge: Optional[KnowledgePrimitive] = None,
        executor: Optional[ExecutorPrimitive] = None
    ):
        """Initialize the DataAgent with configuration and optional components.
        
        Args:
            config: Agent configuration
            llm: Optional LLM interface (will be created from config if not provided)
            knowledge: Optional knowledge primitive (will be created from config if not provided)
            executor: Optional executor primitive (will be created from config if not provided)
        """
        self.config = config
        
        # Initialize components
        self.llm = llm or self._create_llm()
        self.knowledge = knowledge or self._create_knowledge()
        self.executor = executor or self._create_executor()
        
        # Initialize workflow
        self.workflow = DataAgentWorkflow(
            llm=self.llm,
            knowledge=self.knowledge,
            executor=self.executor,
            config=config
        )
        
        # Conversation management
        self._conversations: Dict[str, SharedState] = {}
        
        logger.info(f"DataAgent initialized: {config.name}")
    
    def _create_llm(self) -> LLMInterface:
        """Create LLM interface from configuration.
        
        Returns:
            Configured LLM interface
        """
        return create_llm_interface(self.config.llm)
    
    def _create_knowledge(self) -> KnowledgePrimitive:
        """Create knowledge primitive from configuration.
        
        Returns:
            Configured knowledge primitive
        """
        # For now, only FAISS is implemented
        knowledge_config = self.config.knowledge
        return FAISSKnowledge(
            model_name=knowledge_config.embedding_model,
            index_path=knowledge_config.index_path,
            embedding_dim=getattr(knowledge_config, 'embedding_dim', None)
        )
    
    def _create_executor(self) -> ExecutorPrimitive:
        """Create executor primitive from configuration.
        
        Returns:
            Configured executor primitive
        """
        # For now, only InMemoryExecutor is implemented
        executor_config = self.config.executor
        config_dict = {
            "database_path": getattr(executor_config, 'database_url', ":memory:"),
            "max_execution_time": executor_config.max_execution_time,
            "max_memory_mb": executor_config.max_memory_mb
        }
        return InMemoryExecutor(config_dict)
    
    async def query(
        self,
        query: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """Process a user query and return a response.
        
        Args:
            query: User query to process
            conversation_id: Optional conversation ID for state persistence
            
        Returns:
            Agent response as a string
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Use default conversation ID if not provided
        if conversation_id is None:
            conversation_id = "default"
        
        # Get existing conversation state if available
        existing_state = self._conversations.get(conversation_id)
        
        # Process the query through the workflow
        final_state = await self.workflow.process_query(
            query=query,
            conversation_id=conversation_id,
            existing_state=existing_state
        )
        
        # Store conversation state
        self._conversations[conversation_id] = final_state
        
        # Return the formatted response
        if final_state.formatted_response:
            return final_state.formatted_response
        elif final_state.error_occurred:
            return f"Error: {final_state.error_message}"
        else:
            return "I'm sorry, I couldn't process your query. Please try again."
    
    async def approve_operation(
        self,
        conversation_id: str,
        approved: bool = True,
        reason: Optional[str] = None
    ) -> str:
        """Approve or deny a pending operation.
        
        Args:
            conversation_id: Conversation ID with pending approval
            approved: Whether to approve the operation
            reason: Optional reason for the decision
            
        Returns:
            Response after processing the approval
        """
        logger.info(f"Processing approval for conversation {conversation_id}: {approved}")
        
        # Continue workflow with approval decision
        final_state = await self.workflow.continue_with_approval(
            conversation_id=conversation_id,
            approved=approved,
            reason=reason
        )
        
        # Update conversation state
        self._conversations[conversation_id] = final_state
        
        # Return the response
        if final_state.formatted_response:
            return final_state.formatted_response
        elif final_state.error_occurred:
            return f"Error: {final_state.error_message}"
        else:
            return "Operation processed."
    
    async def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """Get conversation history for a specific conversation.
        
        Args:
            conversation_id: Conversation ID to retrieve
            
        Returns:
            List of messages in the conversation
        """
        state = self._conversations.get(conversation_id)
        if state:
            return state.conversation_history
        return []
    
    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get the current status of a conversation.
        
        Args:
            conversation_id: Conversation ID to check
            
        Returns:
            Dictionary containing conversation status information
        """
        state = self._conversations.get(conversation_id)
        if not state:
            return {"exists": False}
        
        return {
            "exists": True,
            "current_step": state.current_step,
            "workflow_complete": state.workflow_complete,
            "error_occurred": state.error_occurred,
            "error_message": state.error_message,
            "pending_approval": state.pending_approval is not None,
            "iteration_count": state.iteration_count,
            "message_count": len(state.conversation_history)
        }
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation from memory.
        
        Args:
            conversation_id: Conversation ID to clear
            
        Returns:
            True if conversation was cleared, False if it didn't exist
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Cleared conversation: {conversation_id}")
            return True
        return False
    
    async def ingest_knowledge(self, documents: List[Document]) -> None:
        """Ingest documents into the knowledge base.
        
        Args:
            documents: List of documents to ingest
        """
        logger.info(f"Ingesting {len(documents)} documents into knowledge base...")
        await self.knowledge.ingest(documents)
        logger.info("Knowledge ingestion complete")
    
    async def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search the knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional search filters
            
        Returns:
            List of relevant documents
        """
        return await self.knowledge.search(query, limit, filters)
    
    async def get_database_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get database schema information.
        
        Args:
            table_name: Optional specific table name
            
        Returns:
            Schema information
        """
        return await self.executor.get_schema(table_name)
    
    async def list_database_tables(self) -> List[str]:
        """List all available database tables.
        
        Returns:
            List of table names
        """
        return await self.executor.list_tables()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent configuration.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "llm_provider": self.config.llm.provider.value,
            "llm_model": self.config.llm.model,
            "knowledge_provider": self.config.knowledge.provider.value,
            "executor_provider": self.config.executor.provider.value,
            "workflow_info": self.workflow.get_workflow_info(),
            "active_conversations": len(self._conversations)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all components.
        
        Returns:
            Health status of all components
        """
        health = {
            "agent": "healthy",
            "llm": "unknown",
            "knowledge": "unknown",
            "executor": "unknown",
            "timestamp": None
        }
        
        try:
            # Check LLM
            llm_info = await self.llm.get_model_info()
            health["llm"] = "healthy" if llm_info else "unhealthy"
        except Exception as e:
            health["llm"] = f"unhealthy: {e}"
        
        try:
            # Check knowledge base
            kb_stats = await self.knowledge.get_stats()
            health["knowledge"] = "healthy" if kb_stats else "unhealthy"
        except Exception as e:
            health["knowledge"] = f"unhealthy: {e}"
        
        try:
            # Check executor
            tables = await self.executor.list_tables()
            health["executor"] = "healthy" if isinstance(tables, list) else "unhealthy"
        except Exception as e:
            health["executor"] = f"unhealthy: {e}"
        
        from datetime import datetime
        health["timestamp"] = datetime.now().isoformat()
        
        return health
    
    async def shutdown(self) -> None:
        """Shutdown the agent and clean up resources.
        
        This method should be called when the agent is no longer needed
        to ensure proper cleanup of resources.
        """
        logger.info(f"Shutting down DataAgent: {self.config.name}")
        
        # Clear conversations
        self._conversations.clear()
        
        # Note: In a full implementation, we might need to close
        # database connections, save state, etc.
        
        logger.info("DataAgent shutdown complete")


async def create_agent_from_config(config: AgentConfig) -> DataAgent:
    """Factory function to create a DataAgent from configuration.
    
    Args:
        config: Agent configuration
        
    Returns:
        Initialized DataAgent instance
    """
    agent = DataAgent(config)
    
    # Perform initial health check
    health = await agent.health_check()
    logger.info(f"Agent health check: {health}")
    
    return agent