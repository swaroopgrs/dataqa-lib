"""
Tests for the LLM Judge Evaluator with pluggable scoring rubrics.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.dataqa.orchestration.evaluation.judge import (
    LLMJudgeEvaluator,
    ScoringRubric,
    EvaluationResult,
    AgentResponse,
    GroundTruth,
    EvaluationCriterion,
    LLMModel
)
from src.dataqa.config.models import LLMConfig, LLMProvider
from src.dataqa.exceptions import DataQAError


class TestLLMJudgeEvaluator:
    """Test cases for LLMJudgeEvaluator."""
    
    @pytest.fixture
    def llm_configs(self):
        """Create sample LLM configurations."""
        return [
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                api_key="test-key-1",
                temperature=0.1
            ),
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                api_key="test-key-2",
                temperature=0.2
            )
        ]
    
    @pytest.fixture
    def sample_scoring_rubric(self):
        """Create sample scoring rubric."""
        return ScoringRubric(
            name="Test Quality Rubric",
            description="Rubric for testing response quality",
            domain="testing",
            criteria=["accuracy", "completeness", "clarity"],
            scoring_scale={
                "excellent": "90-100: Outstanding quality",
                "good": "70-89: Good quality",
                "acceptable": "50-69: Acceptable quality",
                "poor": "0-49: Poor quality"
            },
            weight_distribution={
                "accuracy": 0.5,
                "completeness": 0.3,
                "clarity": 0.2
            },
            prompt_template="Evaluate based on criteria. Response: {agent_response}. Expected: {ground_truth}. Context: {query_context}."
        )
    
    @pytest.fixture
    def sample_agent_response(self):
        """Create sample agent response."""
        return AgentResponse(
            agent_id="test-agent",
            content={"answer": "The result is 42", "confidence": 0.9},
            execution_time_seconds=2.5,
            metadata={"model": "gpt-4", "tokens": 150}
        )
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth."""
        return GroundTruth(
            expected_output={"answer": "The result is 42", "confidence": 1.0},
            acceptable_variations=[
                {"answer": "42", "confidence": 0.9},
                {"answer": "The answer is 42", "confidence": 0.95}
            ],
            evaluation_notes="Should provide correct numerical answer with confidence"
        )
    
    @pytest.fixture
    def sample_evaluation_criteria(self):
        """Create sample evaluation criteria."""
        return [
            EvaluationCriterion(
                name="accuracy",
                description="Correctness of the answer",
                weight=0.6,
                scoring_method="llm_judge"
            ),
            EvaluationCriterion(
                name="completeness",
                description="Completeness of the response",
                weight=0.4,
                scoring_method="similarity"
            )
        ]
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface."""
        mock_llm = AsyncMock()
        mock_llm.format_response = AsyncMock()
        return mock_llm
    
    @pytest.fixture
    def evaluator_with_mocks(self, llm_configs, mock_llm_interface):
        """Create evaluator with mocked LLM interfaces."""
        with patch('src.dataqa.orchestration.evaluation.judge.create_llm_interface') as mock_create:
            mock_create.return_value = mock_llm_interface
            evaluator = LLMJudgeEvaluator(llm_configs)
            return evaluator, mock_llm_interface
    
    def test_evaluator_initialization(self, llm_configs):
        """Test LLM judge evaluator initialization."""
        with patch('src.dataqa.orchestration.evaluation.judge.create_llm_interface') as mock_create:
            mock_create.return_value = AsyncMock()
            
            evaluator = LLMJudgeEvaluator(llm_configs)
            
            assert len(evaluator.judge_models) == 2
            assert len(evaluator.scoring_rubrics) > 0  # Default rubrics should be loaded
            assert len(evaluator.evaluation_history) == 0
    
    def test_evaluator_initialization_with_failed_models(self, llm_configs):
        """Test evaluator initialization when some models fail to initialize."""
        with patch('src.dataqa.orchestration.evaluation.judge.create_llm_interface') as mock_create:
            # First model succeeds, second fails
            mock_create.side_effect = [AsyncMock(), Exception("Model init failed")]
            
            evaluator = LLMJudgeEvaluator(llm_configs)
            
            assert len(evaluator.judge_models) == 1
            assert len(evaluator.scoring_rubrics) > 0
    
    def test_evaluator_initialization_no_models(self, llm_configs):
        """Test evaluator initialization when no models can be initialized."""
        with patch('src.dataqa.orchestration.evaluation.judge.create_llm_interface') as mock_create:
            mock_create.side_effect = Exception("All models failed")
            
            evaluator = LLMJudgeEvaluator(llm_configs)
            
            assert len(evaluator.judge_models) == 0
            assert len(evaluator.scoring_rubrics) > 0  # Default rubrics still loaded
    
    def test_add_scoring_rubric(self, evaluator_with_mocks, sample_scoring_rubric):
        """Test adding a scoring rubric."""
        evaluator, _ = evaluator_with_mocks
        initial_count = len(evaluator.scoring_rubrics)
        
        evaluator.add_scoring_rubric(sample_scoring_rubric)
        
        assert len(evaluator.scoring_rubrics) == initial_count + 1
        assert sample_scoring_rubric.rubric_id in evaluator.scoring_rubrics
        assert evaluator.scoring_rubrics[sample_scoring_rubric.rubric_id] == sample_scoring_rubric
    
    def test_get_rubric_by_domain(self, evaluator_with_mocks, sample_scoring_rubric):
        """Test getting rubric by domain."""
        evaluator, _ = evaluator_with_mocks
        evaluator.add_scoring_rubric(sample_scoring_rubric)
        
        # Test exact domain match
        rubric = evaluator.get_rubric_by_domain("testing")
        assert rubric == sample_scoring_rubric
        
        # Test fallback to general domain
        rubric = evaluator.get_rubric_by_domain("unknown_domain")
        assert rubric is not None
        assert rubric.domain == "general"
        
        # Test when no rubrics exist (shouldn't happen with default initialization)
        evaluator.scoring_rubrics.clear()
        rubric = evaluator.get_rubric_by_domain("any_domain")
        assert rubric is None
    
    def test_list_rubrics(self, evaluator_with_mocks, sample_scoring_rubric):
        """Test listing all rubrics."""
        evaluator, _ = evaluator_with_mocks
        initial_count = len(evaluator.list_rubrics())
        
        evaluator.add_scoring_rubric(sample_scoring_rubric)
        rubrics = evaluator.list_rubrics()
        
        assert len(rubrics) == initial_count + 1
        assert sample_scoring_rubric in rubrics
    
    @pytest.mark.asyncio
    async def test_evaluate_response_with_llm_judge(
        self, evaluator_with_mocks, sample_agent_response, sample_ground_truth, 
        sample_evaluation_criteria, sample_scoring_rubric
    ):
        """Test response evaluation using LLM judge."""
        evaluator, mock_llm = evaluator_with_mocks
        evaluator.add_scoring_rubric(sample_scoring_rubric)
        
        # Mock LLM response
        mock_evaluation = {
            "overall_score": 85.0,
            "criterion_scores": {"accuracy": 90.0, "completeness": 80.0},
            "explanation": "Good response with minor issues",
            "confidence": 85.0,
            "strengths": ["Correct answer", "Clear format"],
            "weaknesses": ["Could be more detailed"],
            "suggestions": ["Add more context"]
        }
        mock_llm.format_response.return_value = json.dumps(mock_evaluation)
        
        # Evaluate response
        result = await evaluator.evaluate_response(
            sample_agent_response,
            sample_ground_truth,
            sample_evaluation_criteria,
            domain="testing",
            query_context="Test mathematical calculation"
        )
        
        # Verify result
        assert isinstance(result, EvaluationResult)
        assert result.agent_configuration_id == sample_agent_response.agent_id
        assert result.overall_score == 0.85  # Converted to 0-1 scale
        assert result.criterion_scores["accuracy"] == 0.9
        assert result.criterion_scores["completeness"] == 0.8
        assert result.explanation == "Good response with minor issues"
        assert result.confidence_level == 0.85
        assert result.rubric_used == sample_scoring_rubric.name
        assert len(result.strengths) == 2
        assert len(result.weaknesses) == 1
        assert len(result.improvement_suggestions) == 1
        assert result.evaluation_time_seconds > 0
        
        # Verify it's stored in history
        assert len(evaluator.evaluation_history) == 1
        assert evaluator.evaluation_history[0] == result
    
    @pytest.mark.asyncio
    async def test_evaluate_response_with_fallback(
        self, evaluator_with_mocks, sample_agent_response, sample_ground_truth, 
        sample_evaluation_criteria
    ):
        """Test response evaluation with fallback when LLM judge fails."""
        evaluator, mock_llm = evaluator_with_mocks
        
        # Make LLM judge fail
        mock_llm.format_response.side_effect = Exception("LLM failed")
        
        # Evaluate response
        result = await evaluator.evaluate_response(
            sample_agent_response,
            sample_ground_truth,
            sample_evaluation_criteria,
            domain="general"
        )
        
        # Verify fallback result
        assert isinstance(result, EvaluationResult)
        assert result.overall_score == 0.5  # Default fallback score
        assert "LLM evaluation failed" in result.explanation
        assert result.confidence_level == 0.1
    
    @pytest.mark.asyncio
    async def test_evaluate_response_no_judge_models(
        self, llm_configs, sample_agent_response, sample_ground_truth, 
        sample_evaluation_criteria
    ):
        """Test response evaluation when no judge models are available."""
        with patch('src.dataqa.orchestration.evaluation.judge.create_llm_interface') as mock_create:
            mock_create.side_effect = Exception("No models available")
            
            evaluator = LLMJudgeEvaluator(llm_configs)
            
            # Evaluate response
            result = await evaluator.evaluate_response(
                sample_agent_response,
                sample_ground_truth,
                sample_evaluation_criteria
            )
            
            # Should use fallback evaluation
            assert isinstance(result, EvaluationResult)
            assert result.rubric_used == "fallback"
    
    def test_parse_evaluation_response_valid_json(self, evaluator_with_mocks):
        """Test parsing valid JSON evaluation response."""
        evaluator, _ = evaluator_with_mocks
        
        response = json.dumps({
            "overall_score": 85.0,
            "explanation": "Good response",
            "confidence": 80.0
        })
        
        parsed = evaluator._parse_evaluation_response(response)
        
        assert parsed["overall_score"] == 85.0
        assert parsed["explanation"] == "Good response"
        assert parsed["confidence"] == 80.0
    
    def test_parse_evaluation_response_with_markdown(self, evaluator_with_mocks):
        """Test parsing JSON response wrapped in markdown."""
        evaluator, _ = evaluator_with_mocks
        
        response = """
        Here's the evaluation:
        ```json
        {
            "overall_score": 75.0,
            "explanation": "Acceptable response"
        }
        ```
        """
        
        parsed = evaluator._parse_evaluation_response(response)
        
        assert parsed["overall_score"] == 75.0
        assert parsed["explanation"] == "Acceptable response"
    
    def test_parse_evaluation_response_invalid_json(self, evaluator_with_mocks):
        """Test parsing invalid JSON response."""
        evaluator, _ = evaluator_with_mocks
        
        response = "This is not valid JSON"
        
        parsed = evaluator._parse_evaluation_response(response)
        
        # Should return fallback values
        assert parsed["overall_score"] == 50.0
        assert "Failed to parse" in parsed["explanation"]
        assert parsed["confidence"] == 10.0
    
    def test_fallback_evaluate_exact_match(self, evaluator_with_mocks, sample_evaluation_criteria):
        """Test fallback evaluation with exact match."""
        evaluator, _ = evaluator_with_mocks
        
        agent_response = AgentResponse(
            agent_id="test-agent",
            content="exact match"
        )
        ground_truth = GroundTruth(expected_output="exact match")
        
        result = evaluator._fallback_evaluate(
            agent_response, ground_truth, sample_evaluation_criteria
        )
        
        assert result.overall_score == 1.0
        assert result.rubric_used == "fallback"
    
    def test_fallback_evaluate_acceptable_variation(self, evaluator_with_mocks, sample_evaluation_criteria):
        """Test fallback evaluation with acceptable variation."""
        evaluator, _ = evaluator_with_mocks
        
        agent_response = AgentResponse(
            agent_id="test-agent",
            content="variation"
        )
        ground_truth = GroundTruth(
            expected_output="original",
            acceptable_variations=["variation", "alternative"]
        )
        
        result = evaluator._fallback_evaluate(
            agent_response, ground_truth, sample_evaluation_criteria
        )
        
        assert result.overall_score == 0.8
        assert result.rubric_used == "fallback"
    
    def test_fallback_evaluate_no_match(self, evaluator_with_mocks, sample_evaluation_criteria):
        """Test fallback evaluation with no match."""
        evaluator, _ = evaluator_with_mocks
        
        agent_response = AgentResponse(
            agent_id="test-agent",
            content="no match"
        )
        ground_truth = GroundTruth(expected_output="different")
        
        result = evaluator._fallback_evaluate(
            agent_response, ground_truth, sample_evaluation_criteria
        )
        
        assert result.overall_score == 0.5  # Default neutral score
        assert result.rubric_used == "fallback"
    
    @pytest.mark.asyncio
    async def test_generate_explanation(self, evaluator_with_mocks):
        """Test generating detailed explanation for evaluation result."""
        evaluator, _ = evaluator_with_mocks
        
        result = EvaluationResult(
            test_case_id="test-1",
            agent_configuration_id="agent-1",
            overall_score=0.85,
            criterion_scores={"accuracy": 0.9, "completeness": 0.8},
            explanation="Good response overall",
            confidence_level=0.8,
            rubric_used="Test Rubric",
            strengths=["Correct answer", "Clear format"],
            weaknesses=["Missing details"],
            improvement_suggestions=["Add more context", "Improve clarity"]
        )
        
        explanation = await evaluator.generate_explanation(result)
        
        assert "Evaluation Summary (Score: 0.85)" in explanation
        assert "Strengths:" in explanation
        assert "Correct answer" in explanation
        assert "Areas for Improvement:" in explanation
        assert "Missing details" in explanation
        assert "Recommendations:" in explanation
        assert "Add more context" in explanation
        assert "Detailed Scores:" in explanation
        assert "accuracy: 0.90" in explanation
        assert "Confidence Level: 0.80" in explanation
        assert "Rubric Used: Test Rubric" in explanation
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, evaluator_with_mocks):
        """Test calculating confidence level."""
        evaluator, _ = evaluator_with_mocks
        
        result = EvaluationResult(
            test_case_id="test-1",
            agent_configuration_id="agent-1",
            overall_score=0.85,
            confidence_level=0.75,
            explanation="Test result"
        )
        
        confidence = await evaluator.calculate_confidence(result)
        assert confidence == 0.75
    
    def test_get_evaluation_history(self, evaluator_with_mocks):
        """Test getting evaluation history."""
        evaluator, _ = evaluator_with_mocks
        
        # Initially empty
        history = evaluator.get_evaluation_history()
        assert len(history) == 0
        
        # Add some results
        result1 = EvaluationResult(
            test_case_id="test-1",
            agent_configuration_id="agent-1",
            overall_score=0.8,
            confidence_level=0.7,
            explanation="First result"
        )
        result2 = EvaluationResult(
            test_case_id="test-2",
            agent_configuration_id="agent-1",
            overall_score=0.9,
            confidence_level=0.8,
            explanation="Second result"
        )
        
        evaluator.evaluation_history.extend([result1, result2])
        
        # Get history
        history = evaluator.get_evaluation_history()
        assert len(history) == 2
        assert history[0] == result1
        assert history[1] == result2
    
    def test_get_performance_statistics(self, evaluator_with_mocks):
        """Test getting performance statistics."""
        evaluator, _ = evaluator_with_mocks
        
        # Test with empty history
        stats = evaluator.get_performance_statistics()
        assert stats["total_evaluations"] == 0
        
        # Add some evaluation results
        results = [
            EvaluationResult(
                test_case_id=f"test-{i}",
                agent_configuration_id="agent-1",
                overall_score=score,
                confidence_level=0.8,
                explanation=f"Result {i}"
            )
            for i, score in enumerate([0.95, 0.85, 0.75, 0.65, 0.45], 1)
        ]
        
        evaluator.evaluation_history.extend(results)
        
        # Get statistics
        stats = evaluator.get_performance_statistics()
        
        assert stats["total_evaluations"] == 5
        assert stats["average_score"] == 0.73  # (0.95 + 0.85 + 0.75 + 0.65 + 0.45) / 5
        assert stats["average_confidence"] == 0.8
        
        # Check score distribution
        distribution = stats["score_distribution"]
        assert distribution["excellent"] == 1  # >= 0.9 (0.95)
        assert distribution["good"] == 2       # 0.7 <= score < 0.9 (0.85, 0.75)
        assert distribution["acceptable"] == 1  # 0.5 <= score < 0.7 (0.65)
        assert distribution["poor"] == 1       # < 0.5 (0.45)


class TestScoringRubricModels:
    """Test cases for scoring rubric data models."""
    
    def test_scoring_rubric_creation(self):
        """Test ScoringRubric model creation."""
        rubric = ScoringRubric(
            name="Test Rubric",
            description="A test rubric",
            domain="testing",
            criteria=["accuracy", "completeness"],
            scoring_scale={
                "excellent": "90-100",
                "good": "70-89"
            },
            weight_distribution={
                "accuracy": 0.7,
                "completeness": 0.3
            },
            prompt_template="Evaluate: {agent_response}"
        )
        
        assert rubric.name == "Test Rubric"
        assert rubric.description == "A test rubric"
        assert rubric.domain == "testing"
        assert len(rubric.criteria) == 2
        assert rubric.criteria[0] == "accuracy"
        assert rubric.scoring_scale["excellent"] == "90-100"
        assert rubric.weight_distribution["accuracy"] == 0.7
        assert rubric.prompt_template == "Evaluate: {agent_response}"
    
    def test_agent_response_creation(self):
        """Test AgentResponse model creation."""
        response = AgentResponse(
            agent_id="test-agent",
            content={"answer": "42", "reasoning": "calculation"},
            execution_time_seconds=1.5,
            metadata={"model": "gpt-4", "temperature": 0.1}
        )
        
        assert response.agent_id == "test-agent"
        assert response.content["answer"] == "42"
        assert response.execution_time_seconds == 1.5
        assert response.metadata["model"] == "gpt-4"
        assert isinstance(response.timestamp, datetime)
    
    def test_ground_truth_creation(self):
        """Test GroundTruth model creation."""
        ground_truth = GroundTruth(
            expected_output={"answer": "42"},
            acceptable_variations=[{"answer": "forty-two"}],
            evaluation_notes="Should provide numerical answer"
        )
        
        assert ground_truth.expected_output["answer"] == "42"
        assert len(ground_truth.acceptable_variations) == 1
        assert ground_truth.acceptable_variations[0]["answer"] == "forty-two"
        assert ground_truth.evaluation_notes == "Should provide numerical answer"
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult model creation."""
        result = EvaluationResult(
            test_case_id="test-1",
            agent_configuration_id="agent-1",
            overall_score=0.85,
            criterion_scores={"accuracy": 0.9, "completeness": 0.8},
            explanation="Good response with minor issues",
            confidence_level=0.8,
            rubric_used="Test Rubric",
            evaluation_time_seconds=2.5,
            improvement_suggestions=["Add more detail"],
            strengths=["Correct answer"],
            weaknesses=["Lacks context"]
        )
        
        assert result.test_case_id == "test-1"
        assert result.agent_configuration_id == "agent-1"
        assert result.overall_score == 0.85
        assert result.criterion_scores["accuracy"] == 0.9
        assert result.explanation == "Good response with minor issues"
        assert result.confidence_level == 0.8
        assert result.rubric_used == "Test Rubric"
        assert result.evaluation_time_seconds == 2.5
        assert len(result.improvement_suggestions) == 1
        assert len(result.strengths) == 1
        assert len(result.weaknesses) == 1
        assert isinstance(result.timestamp, datetime)
    
    def test_llm_model_creation(self):
        """Test LLMModel model creation."""
        model = LLMModel(
            name="GPT-4 Judge",
            provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=2000,
            specialized_domains=["mathematics", "science"],
            capabilities=["reasoning", "analysis"]
        )
        
        assert model.name == "GPT-4 Judge"
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.temperature == 0.1
        assert model.max_tokens == 2000
        assert "mathematics" in model.specialized_domains
        assert "reasoning" in model.capabilities