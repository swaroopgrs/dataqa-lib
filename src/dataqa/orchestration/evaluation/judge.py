"""
LLM judge evaluator for multi-agent system assessment with pluggable scoring rubrics.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from ...config.models import LLMConfig
from ...primitives.llm import LLMInterface, create_llm_interface
from ...logging_config import get_primitive_logger
from ...exceptions import DataQAError
from ..models import AgentConfiguration


class ScoringRubric(BaseModel):
    """Scoring rubric for domain-specific evaluation."""
    rubric_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    domain: str = "general"
    criteria: List[str] = Field(default_factory=list)
    scoring_scale: Dict[str, str] = Field(default_factory=dict)
    prompt_template: str = ""
    weight_distribution: Dict[str, float] = Field(default_factory=dict)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationCriterion(BaseModel):
    """Individual evaluation criterion."""
    criterion_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    weight: float = 1.0
    scoring_method: str = "llm_judge"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Agent response for evaluation."""
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    content: Any
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GroundTruth(BaseModel):
    """Ground truth for evaluation comparison."""
    truth_id: str = Field(default_factory=lambda: str(uuid4()))
    expected_output: Any
    acceptable_variations: List[Any] = Field(default_factory=list)
    evaluation_notes: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Comprehensive result of LLM judge evaluation."""
    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    test_case_id: str
    agent_configuration_id: str
    overall_score: float
    criterion_scores: Dict[str, float] = Field(default_factory=dict)
    explanation: str
    confidence_level: float
    rubric_used: Optional[str] = None
    evaluation_time_seconds: float = 0.0
    improvement_suggestions: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LLMModel(BaseModel):
    """LLM model configuration for judging."""
    model_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    provider: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2000
    specialized_domains: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMJudgeEvaluator:
    """
    Advanced LLM-based evaluator with pluggable domain-specific scoring rubrics.
    
    Supports multiple judge models, configurable evaluation criteria, and
    structured scoring with reasoning and confidence measures.
    """
    
    def __init__(self, llm_configs: List[LLMConfig]):
        """Initialize the LLM judge evaluator.
        
        Args:
            llm_configs: List of LLM configurations for judge models
        """
        self.logger = get_primitive_logger("judge", "evaluator")
        self.judge_models: List[LLMInterface] = []
        self.scoring_rubrics: Dict[str, ScoringRubric] = {}
        self.evaluation_history: List[EvaluationResult] = []
        
        # Initialize judge models
        for config in llm_configs:
            try:
                judge_model = create_llm_interface(config)
                self.judge_models.append(judge_model)
                self.logger.info(f"Initialized judge model: {config.model}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize judge model {config.model}: {e}")
        
        if not self.judge_models:
            self.logger.warning("No judge models initialized - evaluation will use fallback scoring")
        
        # Initialize default rubrics
        self._initialize_default_rubrics()
    
    def _initialize_default_rubrics(self) -> None:
        """Initialize default scoring rubrics for common domains."""
        
        # General purpose rubric
        general_rubric = ScoringRubric(
            name="General Quality Assessment",
            description="General purpose evaluation rubric for agent responses",
            domain="general",
            criteria=[
                "Correctness and accuracy",
                "Completeness of response",
                "Clarity and coherence",
                "Relevance to query"
            ],
            scoring_scale={
                "excellent": "90-100: Exceptional quality, exceeds expectations",
                "good": "70-89: Good quality with minor issues",
                "acceptable": "50-69: Acceptable but needs improvement",
                "poor": "30-49: Poor quality with significant issues",
                "unacceptable": "0-29: Unacceptable, major problems"
            },
            weight_distribution={
                "correctness": 0.4,
                "completeness": 0.3,
                "clarity": 0.2,
                "relevance": 0.1
            },
            prompt_template="""
            Evaluate the agent response based on the following criteria:
            
            1. Correctness and Accuracy (40%): Is the response factually correct and accurate?
            2. Completeness (30%): Does the response fully address the query?
            3. Clarity and Coherence (20%): Is the response clear and well-structured?
            4. Relevance (10%): Is the response relevant to the original query?
            
            Agent Response: {agent_response}
            Ground Truth: {ground_truth}
            Query Context: {query_context}
            
            Provide your evaluation in the following JSON format:
            {{
                "overall_score": <0-100>,
                "criterion_scores": {{
                    "correctness": <0-100>,
                    "completeness": <0-100>,
                    "clarity": <0-100>,
                    "relevance": <0-100>
                }},
                "explanation": "<detailed explanation>",
                "confidence": <0-100>,
                "strengths": ["<strength1>", "<strength2>"],
                "weaknesses": ["<weakness1>", "<weakness2>"],
                "suggestions": ["<suggestion1>", "<suggestion2>"]
            }}
            """
        )
        self.add_scoring_rubric(general_rubric)
        
        # Data analysis specific rubric
        data_analysis_rubric = ScoringRubric(
            name="Data Analysis Quality",
            description="Specialized rubric for data analysis and SQL query evaluation",
            domain="data_analysis",
            criteria=[
                "Query correctness",
                "Data handling accuracy",
                "Performance efficiency",
                "Business logic alignment",
                "Error handling"
            ],
            scoring_scale={
                "excellent": "90-100: Perfect analysis with optimal approach",
                "good": "70-89: Correct analysis with minor inefficiencies",
                "acceptable": "50-69: Functional but suboptimal approach",
                "poor": "30-49: Incorrect analysis or major issues",
                "unacceptable": "0-29: Completely wrong or dangerous"
            },
            weight_distribution={
                "correctness": 0.35,
                "accuracy": 0.25,
                "efficiency": 0.15,
                "business_logic": 0.15,
                "error_handling": 0.1
            },
            prompt_template="""
            Evaluate this data analysis response focusing on technical accuracy and business value:
            
            1. Query Correctness (35%): Is the SQL/analysis logic correct?
            2. Data Handling Accuracy (25%): Are data types and transformations handled properly?
            3. Performance Efficiency (15%): Is the approach efficient and scalable?
            4. Business Logic Alignment (15%): Does it align with business requirements?
            5. Error Handling (10%): Are edge cases and errors handled appropriately?
            
            Agent Response: {agent_response}
            Expected Output: {ground_truth}
            Business Context: {query_context}
            
            Return evaluation as JSON with scores, explanation, and specific technical feedback.
            """
        )
        self.add_scoring_rubric(data_analysis_rubric)
        
        # Code generation rubric
        code_generation_rubric = ScoringRubric(
            name="Code Generation Quality",
            description="Rubric for evaluating generated code quality and safety",
            domain="code_generation",
            criteria=[
                "Functional correctness",
                "Code quality and style",
                "Security considerations",
                "Performance optimization",
                "Documentation and comments"
            ],
            scoring_scale={
                "excellent": "90-100: Production-ready code with best practices",
                "good": "70-89: Good code with minor style issues",
                "acceptable": "50-69: Functional but needs refinement",
                "poor": "30-49: Works but has significant issues",
                "unacceptable": "0-29: Broken or unsafe code"
            },
            weight_distribution={
                "correctness": 0.4,
                "quality": 0.25,
                "security": 0.2,
                "performance": 0.1,
                "documentation": 0.05
            }
        )
        self.add_scoring_rubric(code_generation_rubric)
    
    def add_scoring_rubric(self, rubric: ScoringRubric) -> None:
        """Add a scoring rubric to the evaluator.
        
        Args:
            rubric: Scoring rubric to add
        """
        self.scoring_rubrics[rubric.rubric_id] = rubric
        self.logger.info(f"Added scoring rubric: {rubric.name} for domain: {rubric.domain}")
    
    def get_rubric_by_domain(self, domain: str) -> Optional[ScoringRubric]:
        """Get the best scoring rubric for a domain.
        
        Args:
            domain: Domain to find rubric for
            
        Returns:
            Best matching scoring rubric or None
        """
        # First try exact domain match
        for rubric in self.scoring_rubrics.values():
            if rubric.domain == domain:
                return rubric
        
        # Fall back to general rubric
        for rubric in self.scoring_rubrics.values():
            if rubric.domain == "general":
                return rubric
        
        return None
    
    def list_rubrics(self) -> List[ScoringRubric]:
        """List all available scoring rubrics.
        
        Returns:
            List of all scoring rubrics
        """
        return list(self.scoring_rubrics.values())
    
    async def evaluate_response(
        self,
        agent_response: AgentResponse,
        ground_truth: GroundTruth,
        evaluation_criteria: List[EvaluationCriterion],
        domain: str = "general",
        query_context: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate an agent response against ground truth using domain-specific rubrics.
        
        Args:
            agent_response: Agent response to evaluate
            ground_truth: Expected ground truth
            evaluation_criteria: Specific evaluation criteria
            domain: Domain for rubric selection
            query_context: Optional context about the original query
            
        Returns:
            Comprehensive evaluation result
            
        Raises:
            DataQAError: If evaluation fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Select appropriate rubric
            rubric = self.get_rubric_by_domain(domain)
            if not rubric:
                self.logger.warning(f"No rubric found for domain: {domain}, using default")
                rubric = self.get_rubric_by_domain("general")
            
            # Choose best judge model for this domain
            judge_model = self._select_judge_model(domain)
            
            if judge_model and rubric:
                # Use LLM judge evaluation
                result = await self._llm_judge_evaluate(
                    agent_response, ground_truth, rubric, judge_model, query_context
                )
            else:
                # Fallback to rule-based evaluation
                result = self._fallback_evaluate(
                    agent_response, ground_truth, evaluation_criteria
                )
            
            # Calculate evaluation time
            evaluation_time = (datetime.utcnow() - start_time).total_seconds()
            result.evaluation_time_seconds = evaluation_time
            
            # Store in history
            self.evaluation_history.append(result)
            
            self.logger.info(f"Evaluated response with score: {result.overall_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise DataQAError(f"Failed to evaluate response: {e}")
    
    def _select_judge_model(self, domain: str) -> Optional[LLMInterface]:
        """Select the best judge model for a domain.
        
        Args:
            domain: Domain to select model for
            
        Returns:
            Best judge model or None
        """
        if not self.judge_models:
            return None
        
        # For now, return the first available model
        # In the future, could implement domain-specific model selection
        return self.judge_models[0]
    
    async def _llm_judge_evaluate(
        self,
        agent_response: AgentResponse,
        ground_truth: GroundTruth,
        rubric: ScoringRubric,
        judge_model: LLMInterface,
        query_context: Optional[str] = None
    ) -> EvaluationResult:
        """Perform LLM-based evaluation using a scoring rubric.
        
        Args:
            agent_response: Agent response to evaluate
            ground_truth: Expected ground truth
            rubric: Scoring rubric to use
            judge_model: LLM model for judging
            query_context: Optional query context
            
        Returns:
            Evaluation result
        """
        # Prepare evaluation prompt
        prompt = rubric.prompt_template.format(
            agent_response=json.dumps(agent_response.content, indent=2, default=str),
            ground_truth=json.dumps(ground_truth.expected_output, indent=2, default=str),
            query_context=query_context or "No additional context provided"
        )
        
        try:
            # Get evaluation from LLM judge
            response = await judge_model.format_response(
                {"evaluation_prompt": prompt},
                "Evaluate the agent response quality"
            )
            
            # Parse JSON response
            evaluation_data = self._parse_evaluation_response(response)
            
            # Create evaluation result
            result = EvaluationResult(
                test_case_id="",  # Will be set by caller
                agent_configuration_id=agent_response.agent_id,
                overall_score=evaluation_data.get("overall_score", 50.0) / 100.0,  # Convert to 0-1 scale
                criterion_scores={
                    k: v / 100.0 for k, v in evaluation_data.get("criterion_scores", {}).items()
                },
                explanation=evaluation_data.get("explanation", "No explanation provided"),
                confidence_level=evaluation_data.get("confidence", 50.0) / 100.0,
                rubric_used=rubric.name,
                improvement_suggestions=evaluation_data.get("suggestions", []),
                strengths=evaluation_data.get("strengths", []),
                weaknesses=evaluation_data.get("weaknesses", [])
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(f"LLM judge evaluation failed: {e}")
            # Return fallback evaluation
            return EvaluationResult(
                test_case_id="",
                agent_configuration_id=agent_response.agent_id,
                overall_score=0.5,
                explanation=f"LLM evaluation failed: {e}",
                confidence_level=0.1,
                rubric_used=rubric.name
            )
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed evaluation data
        """
        try:
            # Try to extract JSON from response
            json_str = response.strip()
            
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON evaluation response")
            return {
                "overall_score": 50.0,
                "explanation": "Failed to parse evaluation response",
                "confidence": 10.0
            }
    
    def _fallback_evaluate(
        self,
        agent_response: AgentResponse,
        ground_truth: GroundTruth,
        evaluation_criteria: List[EvaluationCriterion]
    ) -> EvaluationResult:
        """Fallback rule-based evaluation when LLM judge is unavailable.
        
        Args:
            agent_response: Agent response to evaluate
            ground_truth: Expected ground truth
            evaluation_criteria: Evaluation criteria
            
        Returns:
            Basic evaluation result
        """
        # Simple rule-based evaluation
        score = 0.5  # Default neutral score
        
        # Check for exact match
        if agent_response.content == ground_truth.expected_output:
            score = 1.0
        elif agent_response.content in ground_truth.acceptable_variations:
            score = 0.8
        
        return EvaluationResult(
            test_case_id="",
            agent_configuration_id=agent_response.agent_id,
            overall_score=score,
            explanation="Rule-based evaluation (LLM judge unavailable)",
            confidence_level=0.6,
            rubric_used="fallback"
        )
    
    async def generate_explanation(self, evaluation: EvaluationResult) -> str:
        """Generate detailed explanation for an evaluation result.
        
        Args:
            evaluation: Evaluation result to explain
            
        Returns:
            Detailed explanation string
        """
        explanation = f"Evaluation Summary (Score: {evaluation.overall_score:.2f})\n\n"
        
        if evaluation.strengths:
            explanation += "Strengths:\n"
            for strength in evaluation.strengths:
                explanation += f"• {strength}\n"
            explanation += "\n"
        
        if evaluation.weaknesses:
            explanation += "Areas for Improvement:\n"
            for weakness in evaluation.weaknesses:
                explanation += f"• {weakness}\n"
            explanation += "\n"
        
        if evaluation.improvement_suggestions:
            explanation += "Recommendations:\n"
            for suggestion in evaluation.improvement_suggestions:
                explanation += f"• {suggestion}\n"
            explanation += "\n"
        
        if evaluation.criterion_scores:
            explanation += "Detailed Scores:\n"
            for criterion, score in evaluation.criterion_scores.items():
                explanation += f"• {criterion}: {score:.2f}\n"
        
        explanation += f"\nConfidence Level: {evaluation.confidence_level:.2f}"
        explanation += f"\nRubric Used: {evaluation.rubric_used or 'Default'}"
        
        return explanation
    
    async def calculate_confidence(self, evaluation: EvaluationResult) -> float:
        """Calculate confidence level for an evaluation result.
        
        Args:
            evaluation: Evaluation result
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        # Confidence is already calculated during evaluation
        return evaluation.confidence_level
    
    def get_evaluation_history(self) -> List[EvaluationResult]:
        """Get historical evaluation results.
        
        Returns:
            List of all evaluation results
        """
        return self.evaluation_history.copy()
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the evaluator.
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        scores = [eval_result.overall_score for eval_result in self.evaluation_history]
        confidences = [eval_result.confidence_level for eval_result in self.evaluation_history]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(scores) / len(scores),
            "average_confidence": sum(confidences) / len(confidences),
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 0.9]),
                "good": len([s for s in scores if 0.7 <= s < 0.9]),
                "acceptable": len([s for s in scores if 0.5 <= s < 0.7]),
                "poor": len([s for s in scores if s < 0.5])
            }
        }