"""Workflow nodes for LangGraph-based agent orchestration."""

from typing import Any, Dict

from ..exceptions import ExecutionError, KnowledgeError, LLMError, WorkflowError
from ..logging_config import get_workflow_logger
from ..models.execution import ExecutionResult
from ..primitives.executor import ExecutorPrimitive
from ..primitives.knowledge import KnowledgePrimitive
from ..primitives.llm import LLMInterface
from ..utils.retry import retry_async
from .state import SharedState


class WorkflowNodes:
    """Collection of workflow nodes for agent orchestration.
    
    This class contains all the workflow nodes that can be used in LangGraph
    workflows. Each node is a function that takes the shared state and
    returns an updated state.
    """
    
    def __init__(
        self,
        llm: LLMInterface,
        knowledge: KnowledgePrimitive,
        executor: ExecutorPrimitive,
        require_approval: bool = True
    ):
        """Initialize workflow nodes with required components.
        
        Args:
            llm: LLM interface for code generation and analysis
            knowledge: Knowledge primitive for context retrieval
            executor: Executor primitive for code execution
            require_approval: Whether to require human approval for execution
        """
        self.llm = llm
        self.knowledge = knowledge
        self.executor = executor
        self.require_approval = require_approval
        self.logger = get_workflow_logger("workflow_nodes")
    
    async def query_processor(self, state: SharedState) -> SharedState:
        """Process and analyze the user query.
        
        This node analyzes the user's query to understand intent, extract entities,
        and determine the appropriate approach for generating a response.
        
        Args:
            state: Current shared state
            
        Returns:
            Updated state with query analysis results
        """
        self.logger.info(f"Processing query: {state.current_query[:100]}...")
        
        try:
            # Analyze the query using the LLM
            analysis = await self.llm.analyze_query(
                query=state.current_query,
                conversation_history=state.get_recent_messages()
            )
            
            state.query_analysis = analysis
            state.current_step = "context_retriever"
            
            self.logger.info(f"Query analysis complete: {analysis.get('query_type', 'unknown')}")
            
            # Check if clarification is needed
            if analysis.get("requires_clarification", False):
                # Generate clarification questions
                ambiguities = analysis.get("ambiguities", ["Query intent unclear"])
                clarification = await self.llm.generate_clarification(
                    query=state.current_query,
                    ambiguities=ambiguities,
                    conversation_history=state.get_recent_messages()
                )
                
                state.formatted_response = clarification
                state.workflow_complete = True
                state.current_step = "complete"
                
                self.logger.info("Query requires clarification, workflow complete")
            
        except LLMError as e:
            self.logger.error(f"LLM error in query processing: {e}")
            raise WorkflowError(
                f"Failed to analyze query: {e}",
                user_message="Failed to understand your question. Please try rephrasing it.",
                error_code="QUERY_PROCESSING_FAILED",
                original_error=e
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in query processing: {e}")
            raise WorkflowError(
                f"Unexpected error during query analysis: {e}",
                user_message="An unexpected error occurred while processing your question.",
                error_code="QUERY_PROCESSING_UNEXPECTED_ERROR",
                original_error=e
            )
        
        return state
    
    async def context_retriever(self, state: SharedState) -> SharedState:
        """Retrieve relevant context from the knowledge base.
        
        This node searches the knowledge base for documents relevant to the
        user's query and prepares context for code generation.
        
        Args:
            state: Current shared state
            
        Returns:
            Updated state with retrieved context
        """
        self.logger.info("Retrieving context from knowledge base...")
        
        try:
            # Search for relevant documents
            documents = await self.knowledge.search(
                query=state.current_query,
                limit=5
            )
            
            state.retrieved_context = documents
            
            # Create context summary for LLM
            if documents:
                context_parts = []
                for doc in documents:
                    context_parts.append(f"Source: {doc.source}")
                    context_parts.append(f"Content: {doc.content[:500]}...")
                    if doc.metadata:
                        context_parts.append(f"Metadata: {doc.metadata}")
                    context_parts.append("---")
                
                state.context_summary = "\n".join(context_parts)
                self.logger.info(f"Retrieved {len(documents)} relevant documents")
            else:
                state.context_summary = "No relevant context found in knowledge base."
                self.logger.info("No relevant context found")
            
            state.current_step = "code_generator"
            
        except KnowledgeError as e:
            self.logger.error(f"Knowledge error in context retrieval: {e}")
            # Continue without context rather than failing
            state.retrieved_context = []
            state.context_summary = "Context retrieval failed, proceeding without additional context."
            state.current_step = "code_generator"
            self.logger.warning("Continuing without context due to retrieval error")
        except Exception as e:
            self.logger.error(f"Unexpected error in context retrieval: {e}")
            raise WorkflowError(
                f"Unexpected error during context retrieval: {e}",
                user_message="An error occurred while retrieving relevant information.",
                error_code="CONTEXT_RETRIEVAL_ERROR",
                original_error=e
            )
        
        return state
    
    async def code_generator(self, state: SharedState) -> SharedState:
        """Generate code based on the query and context.
        
        This node uses the LLM to generate appropriate SQL or Python code
        to answer the user's query, incorporating retrieved context.
        
        Args:
            state: Current shared state
            
        Returns:
            Updated state with generated code
        """
        self.logger.info("Generating code for query...")
        
        try:
            # Determine code type based on query analysis
            query_type = state.query_analysis.get("query_type", "analysis") if state.query_analysis else "analysis"
            
            # Default to SQL for data retrieval, Python for analysis/visualization
            if query_type in ["data_retrieval"]:
                code_type = "sql"
            else:
                code_type = "python"
            
            # Generate code using LLM
            generated_code = await self.llm.generate_code(
                query=state.current_query,
                context=state.context_summary,
                code_type=code_type,
                conversation_history=state.get_recent_messages()
            )
            
            state.generated_code = generated_code
            state.code_type = code_type
            
            # Validate the generated code
            validation = await self.llm.validate_generated_code(
                code=generated_code,
                code_type=code_type,
                context=state.context_summary
            )
            
            state.code_validation = validation
            
            self.logger.info(f"Generated {code_type} code, validation: {validation.get('is_valid', False)}")
            
            # Check if code is valid
            if not validation.get("is_valid", False):
                error_msg = f"Generated code failed validation: {validation.get('issues', [])}"
                self.logger.error(error_msg)
                state.set_error(error_msg)
                return state
            
            # Check if approval is required
            risk_level = validation.get("risk_level", "low")
            if self.require_approval and risk_level in ["medium", "high"]:
                state.pending_approval = generated_code
                state.current_step = "approval_gate"
                self.logger.info(f"Code requires approval due to {risk_level} risk level")
            else:
                state.current_step = "executor"
                self.logger.info("Code approved for execution")
            
        except LLMError as e:
            self.logger.error(f"LLM error in code generation: {e}")
            state.set_error(f"Failed to generate code: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in code generation: {e}")
            state.set_error(f"Unexpected error during code generation: {e}")
        
        return state
    
    async def approval_gate(self, state: SharedState) -> SharedState:
        """Handle human-in-the-loop approval for code execution.
        
        This node manages the approval process for potentially risky operations.
        It provides detailed information about the code to be executed and
        handles various approval scenarios with proper error handling.
        
        Args:
            state: Current shared state
            
        Returns:
            Updated state with approval status
        """
        self.logger.info("Processing approval gate...")
        
        try:
            # Validate approval gate state
            if not state.pending_approval:
                self.logger.error("Approval gate called without pending approval")
                state.set_error("No operation pending approval")
                return state
            
            if state.approval_granted:
                # Approval already granted, proceed to execution
                state.current_step = "executor"
                self.logger.info("Approval granted, proceeding to execution")
                
                # Log approval metrics
                self.logger.info("Approval granted")
                state.metadata["approval_granted"] = {
                    "code_type": state.code_type,
                    "risk_level": state.code_validation.get('risk_level', 'unknown') if state.code_validation else 'unknown'
                }
                return state
            
            # Generate comprehensive approval request
            approval_request = await self._generate_approval_request(state)
            
            if not approval_request:
                state.set_error("Failed to generate approval request")
                return state
            
            state.formatted_response = approval_request
            state.workflow_complete = True
            state.current_step = "awaiting_approval"
            
            # Log approval request metrics
            self.logger.info("Approval requested")
            state.metadata["approval_requested"] = {
                "code_type": state.code_type,
                "risk_level": state.code_validation.get('risk_level', 'unknown') if state.code_validation else 'unknown',
                "security_concerns": len(state.code_validation.get('security_concerns', [])) if state.code_validation else 0
            }
            
            self.logger.info("Approval request formatted, workflow paused")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in approval gate: {e}")
            state.set_error(f"Approval gate error: {e}")
            state.metadata["approval_gate_error"] = str(e)
        
        return state
    
    async def _generate_approval_request(self, state: SharedState) -> str:
        """Generate a comprehensive approval request message.
        
        Args:
            state: Current shared state
            
        Returns:
            Formatted approval request string
        """
        try:
            # Extract validation information
            validation = state.code_validation or {}
            risk_level = validation.get('risk_level', 'unknown')
            issues = validation.get('issues', [])
            security_concerns = validation.get('security_concerns', [])
            suggestions = validation.get('suggestions', [])
            
            # Build approval request
            request_parts = [
                "🔒 **CODE APPROVAL REQUIRED**",
                "",
                f"**Query:** {state.current_query}",
                f"**Code Type:** {state.code_type.upper()}",
                f"**Risk Level:** {risk_level.upper()}",
                "",
                "**Code to Execute:**",
                f"```{state.code_type}",
                state.pending_approval,
                "```",
                ""
            ]
            
            # Add validation details if available
            if issues:
                request_parts.extend([
                    "**⚠️ Validation Issues:**",
                    *[f"• {issue}" for issue in issues],
                    ""
                ])
            
            if security_concerns:
                request_parts.extend([
                    "**🛡️ Security Concerns:**",
                    *[f"• {concern}" for concern in security_concerns],
                    ""
                ])
            
            if suggestions:
                request_parts.extend([
                    "**💡 Suggestions:**",
                    *[f"• {suggestion}" for suggestion in suggestions],
                    ""
                ])
            
            # Add risk-specific warnings
            if risk_level == "high":
                request_parts.extend([
                    "**⚠️ HIGH RISK OPERATION**",
                    "This operation has been flagged as high risk. Please review carefully.",
                    ""
                ])
            elif risk_level == "medium":
                request_parts.extend([
                    "**⚠️ MEDIUM RISK OPERATION**",
                    "This operation requires review before execution.",
                    ""
                ])
            
            request_parts.extend([
                "**Please review and approve this code for execution.**",
                "",
                "Reply with 'approve' to execute or 'deny' to cancel."
            ])
            
            return "\n".join(request_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating approval request: {e}")
            # Fallback to simple request
            return f"""
Code Approval Required

Code Type: {state.code_type}
Risk Level: {state.code_validation.get('risk_level', 'unknown') if state.code_validation else 'unknown'}

Code:
```{state.code_type}
{state.pending_approval}
```

Please review and approve this code for execution.
"""
    
    async def execute_code(self, state: SharedState) -> SharedState:
        """Execute the generated code safely.
        
        This node executes the validated code in a secure environment
        and captures the results for response formatting. Includes enhanced
        error handling and recovery mechanisms.
        
        Args:
            state: Current shared state
            
        Returns:
            Updated state with execution results
        """
        self.logger.info(f"Executing {state.code_type} code...")
        
        try:
            if not state.generated_code or not state.code_type:
                state.set_error("No code available for execution")
                return state
            
            # Validate state is ready for execution
            if not state.is_ready_for_execution():
                state.set_error("State not ready for execution - missing validation")
                return state
            
            # Execute the code based on type with timeout and resource limits
            self.logger.info("Starting code execution")
            
            if state.code_type == "sql":
                result = await self.executor.execute_sql(state.generated_code)
            elif state.code_type == "python":
                result = await self.executor.execute_python(state.generated_code)
            else:
                state.set_error(f"Unsupported code type: {state.code_type}")
                return state
            
            state.execution_results = result
            
            if result.success:
                self.logger.info(f"Code execution successful, execution time: {result.execution_time:.2f}s")
                state.current_step = "response_formatter"
                
                # Log execution metrics for monitoring
                state.metadata["execution_metrics"] = {
                    "execution_time": result.execution_time,
                    "code_type": state.code_type,
                    "output_type": result.output_type,
                    "success": True
                }
            else:
                self.logger.error(f"Code execution failed: {result.error}")
                
                # Attempt error recovery based on error type
                recovery_attempted = await self._attempt_execution_recovery(state, result)
                
                if not recovery_attempted:
                    state.set_error(f"Code execution failed: {result.error}")
                    state.metadata["execution_error"] = {
                        "error": result.error,
                        "code_executed": result.code_executed,
                        "execution_time": result.execution_time
                    }
            
        except ExecutionError as e:
            self.logger.error(f"Execution error: {e}")
            state.set_error(f"Failed to execute code: {e}")
            state.metadata["execution_exception"] = str(e)
        except Exception as e:
            self.logger.error(f"Unexpected error in code execution: {e}")
            state.set_error(f"Unexpected error during code execution: {e}")
            state.metadata["unexpected_error"] = str(e)
        
        return state
    
    async def _attempt_execution_recovery(self, state: SharedState, failed_result: ExecutionResult) -> bool:
        """Attempt to recover from execution errors.
        
        Args:
            state: Current shared state
            failed_result: The failed execution result
            
        Returns:
            True if recovery was attempted, False otherwise
        """
        error_msg = failed_result.error or ""
        
        # Common SQL error recovery patterns
        if state.code_type == "sql":
            # Check for column errors first (more specific)
            if "column" in error_msg.lower() and ("not exist" in error_msg.lower() or "does not exist" in error_msg.lower()):
                self.logger.info("Attempting recovery for missing column error")
                state.metadata["recovery_suggestion"] = "Column not found - consider checking table schema"
                return False
            # Then check for table errors
            elif "table" in error_msg.lower() and ("not exist" in error_msg.lower() or "does not exist" in error_msg.lower()):
                self.logger.info("Attempting recovery for missing table error")
                state.metadata["recovery_suggestion"] = "Table not found - consider checking available tables"
                return False
        
        # Common Python error recovery patterns
        elif state.code_type == "python":
            if "no module named" in error_msg.lower():
                self.logger.info("Attempting recovery for missing module error")
                state.metadata["recovery_suggestion"] = "Missing Python module - check available libraries"
                return False
            
            if "name" in error_msg.lower() and "not defined" in error_msg.lower():
                self.logger.info("Attempting recovery for undefined variable error")
                state.metadata["recovery_suggestion"] = "Variable not defined - check variable names"
                return False
        
        return False
    
    async def response_formatter(self, state: SharedState) -> SharedState:
        """Format the execution results into a user-friendly response.
        
        This node takes the raw execution results and formats them into
        a natural language response for the user. Includes enhanced error
        handling and fallback formatting mechanisms.
        
        Args:
            state: Current shared state
            
        Returns:
            Updated state with formatted response
        """
        self.logger.info("Formatting response for user...")
        
        try:
            if not state.execution_results:
                state.set_error("No execution results available for formatting")
                return state
            
            # Prepare results for formatting with enhanced metadata
            results_dict = {
                "success": state.execution_results.success,
                "data": state.execution_results.data,
                "execution_time": state.execution_results.execution_time,
                "output_type": state.execution_results.output_type,
                "code_executed": state.execution_results.code_executed,
                "query": state.current_query,
                "code_type": state.code_type,
                "metadata": state.metadata
            }
            
            # Add recovery suggestions if available
            if "recovery_suggestion" in state.metadata:
                results_dict["recovery_suggestion"] = state.metadata["recovery_suggestion"]
            
            # Format response using LLM with retry logic
            formatted_response = await self._format_response_with_retry(
                results_dict, state
            )
            
            if not formatted_response:
                # Use fallback formatting if LLM formatting fails
                formatted_response = self._generate_fallback_response(state)
            
            state.formatted_response = formatted_response
            state.workflow_complete = True
            state.current_step = "complete"
            
            # Log formatting metrics
            self.logger.info("Response formatting complete")
            state.metadata["response_formatted"] = {
                "response_length": len(formatted_response),
                "fallback_used": "fallback" in formatted_response.lower()
            }
            
            self.logger.info("Response formatting complete")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in response formatting: {e}")
            
            # Generate emergency fallback response
            try:
                fallback_response = self._generate_fallback_response(state)
                state.formatted_response = fallback_response
                state.workflow_complete = True
                state.current_step = "complete"
                state.metadata["emergency_fallback"] = True
                self.logger.warning("Using emergency fallback response")
            except Exception as fallback_error:
                self.logger.error(f"Emergency fallback failed: {fallback_error}")
                state.set_error(f"Response formatting failed: {e}")
        
        return state
    
    async def _format_response_with_retry(self, results_dict: dict, state: SharedState, max_retries: int = 2) -> str:
        """Format response with retry logic for LLM failures.
        
        Args:
            results_dict: Results dictionary for formatting
            state: Current shared state
            max_retries: Maximum number of retry attempts
            
        Returns:
            Formatted response string or empty string if all attempts fail
        """
        for attempt in range(max_retries + 1):
            try:
                formatted_response = await self.llm.format_response(
                    results=results_dict,
                    query=state.current_query,
                    conversation_history=state.get_recent_messages()
                )
                return formatted_response
                
            except LLMError as e:
                self.logger.warning(f"LLM formatting attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    self.logger.error("All LLM formatting attempts failed")
                    return ""
                # Wait briefly before retry (in a real implementation)
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error in formatting attempt {attempt + 1}: {e}")
                return ""
        
        return ""
    
    def _generate_fallback_response(self, state: SharedState) -> str:
        """Generate a fallback response when LLM formatting fails.
        
        Args:
            state: Current shared state
            
        Returns:
            Fallback formatted response
        """
        try:
            if not state.execution_results:
                return "Query processing completed, but no results are available."
            
            result = state.execution_results
            
            # Build fallback response based on execution results
            response_parts = [
                f"✅ **Query Executed Successfully**",
                "",
                f"**Query:** {state.current_query}",
                f"**Execution Time:** {result.execution_time:.2f} seconds",
                f"**Code Type:** {state.code_type.upper() if state.code_type else 'Unknown'}",
                ""
            ]
            
            # Add data summary
            if result.data:
                if isinstance(result.data, dict):
                    if "columns" in result.data and "data" in result.data:
                        # DataFrame-like structure
                        num_rows = len(result.data["data"]) if result.data["data"] else 0
                        num_cols = len(result.data["columns"]) if result.data["columns"] else 0
                        response_parts.extend([
                            f"**Results:** {num_rows} rows, {num_cols} columns",
                            ""
                        ])
                    else:
                        # Generic dictionary
                        response_parts.extend([
                            f"**Results:** {len(result.data)} data items",
                            ""
                        ])
                elif isinstance(result.data, list):
                    response_parts.extend([
                        f"**Results:** {len(result.data)} items",
                        ""
                    ])
                else:
                    response_parts.extend([
                        "**Results:** Data available",
                        ""
                    ])
            
            # Add output type information
            if result.output_type:
                response_parts.extend([
                    f"**Output Type:** {result.output_type.title()}",
                    ""
                ])
            
            # Add recovery suggestions if available
            if "recovery_suggestion" in state.metadata:
                response_parts.extend([
                    "**Note:** " + state.metadata["recovery_suggestion"],
                    ""
                ])
            
            # Add code executed (truncated)
            if result.code_executed:
                code_preview = result.code_executed[:200]
                if len(result.code_executed) > 200:
                    code_preview += "..."
                
                response_parts.extend([
                    "**Code Executed:**",
                    f"```{state.code_type or 'text'}",
                    code_preview,
                    "```"
                ])
            
            return "\n".join(response_parts)
            
        except Exception as e:
            self.logger.error(f"Fallback response generation failed: {e}")
            return f"Query completed. Execution time: {state.execution_results.execution_time:.2f}s" if state.execution_results else "Query completed."
    
    def grant_approval(self, state: SharedState) -> SharedState:
        """Grant approval for pending operations.
        
        This is a utility method that can be called externally to grant
        approval for operations waiting in the approval gate.
        
        Args:
            state: Current shared state
            
        Returns:
            Updated state with approval granted
        """
        if state.pending_approval:
            state.approval_granted = True
            state.current_step = "executor"
            self.logger.info("Approval granted for pending operation")
        else:
            self.logger.warning("No pending approval to grant")
        
        return state
    
    def deny_approval(self, state: SharedState, reason: str = "User denied approval") -> SharedState:
        """Deny approval for pending operations.
        
        Args:
            state: Current shared state
            reason: Reason for denial
            
        Returns:
            Updated state with approval denied
        """
        if state.pending_approval:
            state.set_error(f"Operation denied: {reason}")
            logger.info(f"Approval denied: {reason}")
        else:
            logger.warning("No pending approval to deny")
        
        return state