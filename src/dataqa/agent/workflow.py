"""LangGraph workflow implementation for DataQA agent orchestration."""

from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..config.models import AgentConfig
from ..exceptions import WorkflowError
from ..logging_config import get_workflow_logger
from ..primitives.executor import ExecutorPrimitive
from ..primitives.knowledge import KnowledgePrimitive
from ..primitives.llm import LLMInterface
from ..utils.retry import retry_async
from .nodes import WorkflowNodes
from .state import SharedState


class DataAgentWorkflow:
    """LangGraph-based workflow for DataQA agent orchestration.
    
    This class implements the main agent workflow using LangGraph's state
    management and graph execution capabilities. It coordinates between
    query processing, context retrieval, code generation, and execution.
    """
    
    def __init__(
        self,
        llm: LLMInterface,
        knowledge: KnowledgePrimitive,
        executor: ExecutorPrimitive,
        config: AgentConfig
    ):
        """Initialize the workflow with required components.
        
        Args:
            llm: LLM interface for code generation and analysis
            knowledge: Knowledge primitive for context retrieval
            executor: Executor primitive for code execution
            config: Agent configuration
        """
        self.llm = llm
        self.knowledge = knowledge
        self.executor = executor
        self.config = config
        
        # Initialize workflow nodes
        self.nodes = WorkflowNodes(
            llm=llm,
            knowledge=knowledge,
            executor=executor,
            require_approval=config.workflow.require_approval
        )
        
        # Initialize memory for conversation persistence
        self.memory = MemorySaver()
        
        # Initialize logger
        self.logger = get_workflow_logger(config.name)
        
        # Build the workflow graph
        self.graph = self._build_graph()
        
        self.logger.info(f"DataAgent workflow initialized: {config.name}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph.
        
        Returns:
            Configured StateGraph for workflow execution
        """
        # Create the graph with SharedState
        workflow = StateGraph(SharedState)
        
        # Add nodes
        workflow.add_node("query_processor", self.nodes.query_processor)
        workflow.add_node("context_retriever", self.nodes.context_retriever)
        workflow.add_node("code_generator", self.nodes.code_generator)
        workflow.add_node("approval_gate", self.nodes.approval_gate)
        workflow.add_node("executor", self.nodes.execute_code)
        workflow.add_node("response_formatter", self.nodes.response_formatter)
        
        # Set entry point
        workflow.set_entry_point("query_processor")
        
        # Add conditional edges based on state
        workflow.add_conditional_edges(
            "query_processor",
            self._route_from_query_processor,
            {
                "context_retriever": "context_retriever",
                "complete": END
            }
        )
        
        workflow.add_edge("context_retriever", "code_generator")
        
        workflow.add_conditional_edges(
            "code_generator",
            self._route_from_code_generator,
            {
                "approval_gate": "approval_gate",
                "executor": "executor",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "approval_gate",
            self._route_from_approval_gate,
            {
                "executor": "executor",
                "awaiting_approval": END,
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "executor",
            self._route_from_executor,
            {
                "response_formatter": "response_formatter",
                "error": END
            }
        )
        
        workflow.add_edge("response_formatter", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _route_from_query_processor(self, state: SharedState) -> str:
        """Route from query processor based on state.
        
        Args:
            state: Current shared state
            
        Returns:
            Next node to execute
        """
        if state.workflow_complete:
            return "complete"
        elif state.error_occurred:
            return "complete"
        else:
            return "context_retriever"
    
    def _route_from_code_generator(self, state: SharedState) -> str:
        """Route from code generator based on state.
        
        Args:
            state: Current shared state
            
        Returns:
            Next node to execute
        """
        if state.error_occurred:
            return "error"
        elif state.pending_approval:
            return "approval_gate"
        else:
            return "executor"
    
    def _route_from_approval_gate(self, state: SharedState) -> str:
        """Route from approval gate based on state.
        
        Args:
            state: Current shared state
            
        Returns:
            Next node to execute
        """
        if state.error_occurred:
            return "error"
        elif state.workflow_complete:
            return "awaiting_approval"
        elif state.approval_granted:
            return "executor"
        else:
            return "awaiting_approval"
    
    def _route_from_executor(self, state: SharedState) -> str:
        """Route from executor based on state.
        
        Args:
            state: Current shared state
            
        Returns:
            Next node to execute
        """
        if state.error_occurred:
            return "error"
        else:
            return "response_formatter"
    
    async def process_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        existing_state: Optional[SharedState] = None
    ) -> SharedState:
        """Process a user query through the workflow.
        
        Args:
            query: User query to process
            conversation_id: Optional conversation ID for state persistence
            existing_state: Optional existing state to continue from
            
        Returns:
            Final state after workflow execution
        """
        self.logger.info(f"Processing query: {query[:100]}...")
        
        # Initialize or update state
        if existing_state:
            state = existing_state
            state.reset_for_new_query(query)
        else:
            state = SharedState(
                current_query=query,
                max_iterations=self.config.workflow.max_iterations
            )
        
        # Add user message to conversation history
        state.add_message("user", query)
        
        # Configure execution
        config = {
            "configurable": {
                "thread_id": conversation_id or "default"
            }
        }
        
        try:
            # Execute the workflow
            result = await self.graph.ainvoke(
                state,
                config=config
            )
            
            # LangGraph returns a dictionary representation of the state
            # We need to reconstruct the SharedState object
            if isinstance(result, dict):
                final_state = SharedState.model_validate(result)
            else:
                final_state = result
            
            # Add assistant response to conversation history
            if final_state.formatted_response:
                final_state.add_message("assistant", final_state.formatted_response)
            
            self.logger.info(f"Query processing complete: {final_state.current_step}")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise WorkflowError(
                f"Workflow execution failed: {e}",
                user_message="An error occurred while processing your request. Please try again.",
                error_code="WORKFLOW_EXECUTION_FAILED",
                original_error=e
            )
    
    async def continue_with_approval(
        self,
        conversation_id: str,
        approved: bool,
        reason: Optional[str] = None
    ) -> SharedState:
        """Continue workflow execution after approval decision.
        
        Args:
            conversation_id: Conversation ID to continue
            approved: Whether the operation was approved
            reason: Optional reason for approval/denial
            
        Returns:
            Updated state after approval processing
        """
        self.logger.info(f"Continuing workflow with approval: {approved}")
        
        # Get current state from memory
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        try:
            # Get the current state
            current_state = await self.graph.aget_state(config)
            if not current_state or not current_state.values:
                raise ValueError("No conversation state found")
            
            state = current_state.values["state"]
            
            # Process approval decision
            if approved:
                self.nodes.grant_approval(state)
            else:
                self.nodes.deny_approval(state, reason or "User denied approval")
            
            # Continue execution if approved
            if approved and not state.error_occurred:
                final_state = await self.graph.ainvoke(
                    state,
                    config=config
                )
                
                # Add assistant response to conversation history
                if final_state.formatted_response:
                    final_state.add_message("assistant", final_state.formatted_response)
                
                return final_state
            else:
                return state
                
        except Exception as e:
            self.logger.error(f"Approval continuation failed: {e}")
            # Create error state
            error_state = SharedState()
            error_state.set_error(f"Approval continuation failed: {e}")
            return error_state
    
    async def get_conversation_state(self, conversation_id: str) -> Optional[SharedState]:
        """Get the current conversation state.
        
        Args:
            conversation_id: Conversation ID to retrieve
            
        Returns:
            Current conversation state or None if not found
        """
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        try:
            current_state = await self.graph.aget_state(config)
            if current_state and current_state.values:
                return current_state.values["state"]
            return None
        except Exception as e:
            self.logger.error(f"Failed to get conversation state: {e}")
            return None
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow configuration.
        
        Returns:
            Dictionary containing workflow information
        """
        return {
            "agent_name": self.config.name,
            "workflow_strategy": self.config.workflow.strategy,
            "max_iterations": self.config.workflow.max_iterations,
            "require_approval": self.config.workflow.require_approval,
            "auto_approve_safe": self.config.workflow.auto_approve_safe,
            "conversation_memory": self.config.workflow.conversation_memory,
            "enable_visualization": self.config.workflow.enable_visualization,
            "nodes": [
                "query_processor",
                "context_retriever", 
                "code_generator",
                "approval_gate",
                "executor",
                "response_formatter"
            ]
        }