"""
Tests for business rules engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.orchestration.domain.rules import (
    BusinessRulesEngine,
    ComplianceReport,
    ConstraintValidator,
    ExpressionRuleValidator,
    PolicyRuleValidator,
    ValidationResult
)
from src.dataqa.orchestration.models import (
    BusinessRule,
    DomainContext,
    Policy,
    RegulatoryRequirement,
    SchemaConstraint
)


class TestValidationResult:
    """Test ValidationResult model."""
    
    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            is_valid=True,
            violations=["violation1"],
            warnings=["warning1"],
            applied_rules=["rule1"],
            compliance_score=0.8
        )
        
        assert result.is_valid is True
        assert result.violations == ["violation1"]
        assert result.warnings == ["warning1"]
        assert result.applied_rules == ["rule1"]
        assert result.compliance_score == 0.8
    
    def test_validation_result_defaults(self):
        """Test validation result with default values."""
        result = ValidationResult(is_valid=False)
        
        assert result.is_valid is False
        assert result.violations == []
        assert result.warnings == []
        assert result.applied_rules == []
        assert result.compliance_score == 1.0


class TestComplianceReport:
    """Test ComplianceReport model."""
    
    def test_compliance_report_creation(self):
        """Test creating a compliance report."""
        result = ValidationResult(is_valid=True)
        report = ComplianceReport(
            domain_name="finance",
            overall_compliance=True,
            compliance_score=0.95,
            rule_results=[result],
            recommendations=["recommendation1"]
        )
        
        assert report.domain_name == "finance"
        assert report.overall_compliance is True
        assert report.compliance_score == 0.95
        assert len(report.rule_results) == 1
        assert report.recommendations == ["recommendation1"]


class TestExpressionRuleValidator:
    """Test ExpressionRuleValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create an expression rule validator for testing."""
        return ExpressionRuleValidator()
    
    @pytest.fixture
    def sample_rule(self):
        """Create a sample business rule for testing."""
        return BusinessRule(
            rule_id="rule_001",
            name="Test Rule",
            description="A test rule",
            rule_type="expression",
            condition="has_field('user_id')",
            action="validate"
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample domain context for testing."""
        return DomainContext(domain_name="test")
    
    def test_supports_rule_type(self, validator):
        """Test rule type support checking."""
        assert validator.supports_rule_type("expression")
        assert validator.supports_rule_type("condition")
        assert validator.supports_rule_type("constraint")
        assert not validator.supports_rule_type("policy")
    
    @pytest.mark.asyncio
    async def test_validate_has_field_success(self, validator, sample_rule, sample_context):
        """Test successful validation with has_field expression."""
        action = {"user_id": "123", "amount": 100}
        sample_rule.condition = "has_field('user_id')"
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert "rule_001" in result.applied_rules
    
    @pytest.mark.asyncio
    async def test_validate_has_field_failure(self, validator, sample_rule, sample_context):
        """Test failed validation with has_field expression."""
        action = {"amount": 100}
        sample_rule.condition = "has_field('user_id')"
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is False
        assert len(result.violations) == 1
        assert "Rule violation: Test Rule" in result.violations[0]
        assert result.compliance_score == 0.0
    
    @pytest.mark.asyncio
    async def test_validate_field_equals_success(self, validator, sample_rule, sample_context):
        """Test successful validation with field_equals expression."""
        action = {"status": "active", "user_id": "123"}
        sample_rule.condition = "field_equals('status', 'active')"
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_validate_field_equals_failure(self, validator, sample_rule, sample_context):
        """Test failed validation with field_equals expression."""
        action = {"status": "inactive", "user_id": "123"}
        sample_rule.condition = "field_equals('status', 'active')"
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is False
        assert len(result.violations) == 1
    
    @pytest.mark.asyncio
    async def test_validate_field_greater_than_success(self, validator, sample_rule, sample_context):
        """Test successful validation with field_greater_than expression."""
        action = {"amount": 150.5, "user_id": "123"}
        sample_rule.condition = "field_greater_than('amount', 100)"
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_validate_field_greater_than_failure(self, validator, sample_rule, sample_context):
        """Test failed validation with field_greater_than expression."""
        action = {"amount": 50, "user_id": "123"}
        sample_rule.condition = "field_greater_than('amount', 100)"
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is False
        assert len(result.violations) == 1
    
    @pytest.mark.asyncio
    async def test_validate_unknown_expression(self, validator, sample_rule, sample_context):
        """Test validation with unknown expression (should default to True)."""
        action = {"user_id": "123"}
        sample_rule.condition = "unknown_expression('test')"
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_validate_expression_error(self, validator, sample_rule, sample_context):
        """Test handling of expression evaluation errors."""
        action = {"user_id": "123"}
        sample_rule.condition = "invalid_expression(("
        
        # Mock the _evaluate_expression method to raise an exception
        original_method = validator._evaluate_expression
        validator._evaluate_expression = MagicMock(side_effect=Exception("Test error"))
        
        try:
            result = await validator.validate(action, sample_rule, sample_context)
            
            assert result.is_valid is False
            assert len(result.violations) == 1
            assert "Rule evaluation error" in result.violations[0]
            assert result.compliance_score == 0.0
        finally:
            validator._evaluate_expression = original_method


class TestPolicyRuleValidator:
    """Test PolicyRuleValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a policy rule validator for testing."""
        return PolicyRuleValidator()
    
    @pytest.fixture
    def sample_policy(self):
        """Create a sample policy for testing."""
        return Policy(
            policy_id="policy_001",
            name="Data Access Policy",
            description="Controls data access",
            policy_text="Users must have explicit permission for data access",
            enforcement_level="mandatory"
        )
    
    @pytest.fixture
    def sample_rule(self):
        """Create a sample business rule for testing."""
        return BusinessRule(
            rule_id="policy_001",
            name="Data Access Policy",
            description="Policy-based rule",
            rule_type="policy",
            condition="policy_check",
            action="enforce"
        )
    
    @pytest.fixture
    def sample_context(self, sample_policy):
        """Create a sample domain context with policies."""
        return DomainContext(
            domain_name="test",
            organizational_policies=[sample_policy]
        )
    
    def test_supports_rule_type(self, validator):
        """Test rule type support checking."""
        assert validator.supports_rule_type("policy")
        assert validator.supports_rule_type("organizational_policy")
        assert not validator.supports_rule_type("expression")
    
    @pytest.mark.asyncio
    async def test_validate_mandatory_policy_success(self, validator, sample_rule, sample_context):
        """Test successful validation of mandatory policy."""
        action = {"type": "data_access", "has_approval": True}
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert result.metadata["enforcement_level"] == "mandatory"
    
    @pytest.mark.asyncio
    async def test_validate_mandatory_policy_failure(self, validator, sample_rule, sample_context):
        """Test failed validation of mandatory policy."""
        action = {"type": "data_access", "has_approval": False}
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is False
        assert len(result.violations) == 1
        assert "Policy violation" in result.violations[0]
        assert result.compliance_score == 0.0
    
    @pytest.mark.asyncio
    async def test_validate_recommended_policy_failure(self, validator, sample_rule, sample_context):
        """Test failed validation of recommended policy (should be warning)."""
        # Change policy to recommended
        sample_context.organizational_policies[0].enforcement_level = "recommended"
        action = {"type": "data_access", "has_approval": False}
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True  # Recommended policies don't fail validation
        assert len(result.warnings) == 1
        assert "Policy recommendation" in result.warnings[0]
        assert result.compliance_score == 0.5
    
    @pytest.mark.asyncio
    async def test_validate_optional_policy(self, validator, sample_rule, sample_context):
        """Test validation of optional policy."""
        # Change policy to optional
        sample_context.organizational_policies[0].enforcement_level = "optional"
        action = {"type": "data_access", "has_approval": False}
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 0
        assert result.metadata["enforcement_level"] == "optional"
    
    @pytest.mark.asyncio
    async def test_validate_policy_not_found(self, validator, sample_rule, sample_context):
        """Test validation when policy is not found."""
        sample_rule.name = "Nonexistent Policy"
        sample_rule.rule_id = "nonexistent"
        action = {"type": "data_access"}
        
        result = await validator.validate(action, sample_rule, sample_context)
        
        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "Policy not found" in result.warnings[0]
    
    def test_validate_policy_compliance_approval_required(self, validator):
        """Test policy compliance validation for approval requirements."""
        action = {"type": "data_access", "has_approval": True}
        policy = Policy(
            name="Test Policy",
            description="Test policy for approval requirements",
            policy_text="approval_required for all operations",
            enforcement_level="mandatory"
        )
        
        result = validator._validate_policy_compliance(action, policy)
        assert result is True
        
        action["has_approval"] = False
        result = validator._validate_policy_compliance(action, policy)
        assert result is False
    
    def test_validate_policy_compliance_data_access(self, validator):
        """Test policy compliance validation for data access."""
        action = {"type": "data_access", "data_access_level": "read"}
        policy = Policy(
            name="Test Policy",
            description="Test policy for data access restrictions",
            policy_text="data_access restrictions apply",
            enforcement_level="mandatory"
        )
        
        result = validator._validate_policy_compliance(action, policy)
        assert result is True
        
        action["data_access_level"] = "write"
        result = validator._validate_policy_compliance(action, policy)
        assert result is False


class TestConstraintValidator:
    """Test ConstraintValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a constraint validator for testing."""
        return ConstraintValidator()
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample schema constraints for testing."""
        return [
            SchemaConstraint(
                constraint_id="c1",
                name="Required Field",
                schema_path="$.user_id",
                constraint_type="required",
                constraint_value=True,
                error_message="User ID is required"
            ),
            SchemaConstraint(
                constraint_id="c2",
                name="Type Check",
                schema_path="$.amount",
                constraint_type="type",
                constraint_value="number",
                error_message="Amount must be a number"
            ),
            SchemaConstraint(
                constraint_id="c3",
                name="Range Check",
                schema_path="$.amount",
                constraint_type="range",
                constraint_value={"min": 0, "max": 1000},
                error_message="Amount must be between 0 and 1000"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_validate_constraints_success(self, validator, sample_constraints):
        """Test successful constraint validation."""
        action = {"user_id": "123", "amount": 500}
        
        result = await validator.validate_constraints(action, sample_constraints)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert len(result.applied_rules) == 3
        assert result.compliance_score == 1.0
    
    @pytest.mark.asyncio
    async def test_validate_constraints_required_failure(self, validator, sample_constraints):
        """Test constraint validation with required field missing."""
        action = {"amount": 500}  # Missing user_id
        
        result = await validator.validate_constraints(action, sample_constraints)
        
        assert result.is_valid is False
        assert len(result.violations) == 1
        assert "User ID is required" in result.violations[0]
        assert result.compliance_score < 1.0
    
    @pytest.mark.asyncio
    async def test_validate_constraints_type_failure(self, validator, sample_constraints):
        """Test constraint validation with type mismatch."""
        action = {"user_id": "123", "amount": "not_a_number"}
        
        result = await validator.validate_constraints(action, sample_constraints)
        
        assert result.is_valid is False
        assert len(result.violations) == 1
        assert "Amount must be a number" in result.violations[0]
    
    @pytest.mark.asyncio
    async def test_validate_constraints_range_failure(self, validator, sample_constraints):
        """Test constraint validation with range violation."""
        action = {"user_id": "123", "amount": 1500}  # Exceeds max
        
        result = await validator.validate_constraints(action, sample_constraints)
        
        assert result.is_valid is False
        assert len(result.violations) == 1
        assert "Amount must be between 0 and 1000" in result.violations[0]
    
    @pytest.mark.asyncio
    async def test_validate_constraints_warnings(self, validator):
        """Test constraint validation with warnings."""
        constraints = [
            SchemaConstraint(
                constraint_id="w1",
                name="Warning Constraint",
                schema_path="$.optional_field",
                constraint_type="required",
                constraint_value=True,
                error_message="Optional field is missing",
                severity="warning"
            )
        ]
        
        action = {"user_id": "123"}  # Missing optional_field
        
        result = await validator.validate_constraints(action, constraints)
        
        assert result.is_valid is True  # Warnings don't fail validation
        assert len(result.warnings) == 1
        assert "Optional field is missing" in result.warnings[0]
    
    def test_validate_constraint_required(self, validator):
        """Test required constraint validation."""
        constraint = SchemaConstraint(
            name="Test",
            schema_path="$.field",
            constraint_type="required",
            constraint_value=True,
            error_message="Field required"
        )
        
        # Test with field present
        assert validator._validate_constraint({"field": "value"}, constraint) is True
        
        # Test with field missing
        assert validator._validate_constraint({}, constraint) is False
        
        # Test with field None
        assert validator._validate_constraint({"field": None}, constraint) is False
    
    def test_validate_constraint_type(self, validator):
        """Test type constraint validation."""
        constraint = SchemaConstraint(
            name="Test",
            schema_path="$.field",
            constraint_type="type",
            constraint_value="string",
            error_message="Must be string"
        )
        
        assert validator._validate_constraint({"field": "text"}, constraint) is True
        assert validator._validate_constraint({"field": 123}, constraint) is False
        
        # Test number type
        constraint.constraint_value = "number"
        assert validator._validate_constraint({"field": 123}, constraint) is True
        assert validator._validate_constraint({"field": 123.5}, constraint) is True
        assert validator._validate_constraint({"field": "text"}, constraint) is False
    
    def test_validate_constraint_range(self, validator):
        """Test range constraint validation."""
        constraint = SchemaConstraint(
            name="Test",
            schema_path="$.field",
            constraint_type="range",
            constraint_value={"min": 10, "max": 100},
            error_message="Out of range"
        )
        
        assert validator._validate_constraint({"field": 50}, constraint) is True
        assert validator._validate_constraint({"field": 10}, constraint) is True
        assert validator._validate_constraint({"field": 100}, constraint) is True
        assert validator._validate_constraint({"field": 5}, constraint) is False
        assert validator._validate_constraint({"field": 150}, constraint) is False
    
    def test_extract_value_simple(self, validator):
        """Test simple value extraction."""
        data = {"field": "value", "nested": {"inner": "inner_value"}}
        
        assert validator._extract_value(data, "$") == data
        assert validator._extract_value(data, "$.field") == "value"
        assert validator._extract_value(data, "$.nested.inner") == "inner_value"
        assert validator._extract_value(data, "$.nonexistent") is None


class TestBusinessRulesEngine:
    """Test BusinessRulesEngine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create a business rules engine for testing."""
        return BusinessRulesEngine()
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample domain context for testing."""
        rules = [
            BusinessRule(
                rule_id="rule_001",
                name="Expression Rule",
                description="Test expression rule",
                rule_type="expression",
                condition="has_field('user_id')",
                action="validate",
                enabled=True
            ),
            BusinessRule(
                rule_id="rule_002",
                name="Policy Rule",
                description="Test policy rule",
                rule_type="policy",
                condition="policy_check",
                action="enforce",
                enabled=True
            )
        ]
        
        constraints = [
            SchemaConstraint(
                constraint_id="c1",
                name="Required User ID",
                schema_path="$.user_id",
                constraint_type="required",
                constraint_value=True,
                error_message="User ID is required"
            )
        ]
        
        policies = [
            Policy(
                policy_id="policy_001",
                name="Policy Rule",
                description="Test policy",
                policy_text="approval_required for operations",
                enforcement_level="mandatory"
            )
        ]
        
        return DomainContext(
            domain_name="test",
            applicable_rules=rules,
            schema_constraints=constraints,
            organizational_policies=policies
        )
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert len(engine.rule_validators) == 2  # Default validators
        assert isinstance(engine.constraint_validator, ConstraintValidator)
        assert isinstance(engine.rule_cache, dict)
    
    def test_register_validator(self, engine):
        """Test registering a custom validator."""
        custom_validator = ExpressionRuleValidator()
        initial_count = len(engine.rule_validators)
        
        engine.register_validator(custom_validator)
        
        assert len(engine.rule_validators) == initial_count + 1
        assert custom_validator in engine.rule_validators
    
    def test_find_validator(self, engine):
        """Test finding validators by rule type."""
        validator = engine._find_validator("expression")
        assert validator is not None
        assert isinstance(validator, ExpressionRuleValidator)
        
        validator = engine._find_validator("policy")
        assert validator is not None
        assert isinstance(validator, PolicyRuleValidator)
        
        validator = engine._find_validator("unknown_type")
        assert validator is None
    
    @pytest.mark.asyncio
    async def test_validate_action_success(self, engine, sample_context):
        """Test successful action validation."""
        action = {"user_id": "123", "type": "data_access", "has_approval": True}
        
        result = await engine.validate_action(action, sample_context)
        
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert len(result.applied_rules) > 0
        assert result.compliance_score > 0
    
    @pytest.mark.asyncio
    async def test_validate_action_failure(self, engine, sample_context):
        """Test failed action validation."""
        action = {"type": "data_access", "has_approval": False}  # Missing user_id
        
        result = await engine.validate_action(action, sample_context)
        
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert result.compliance_score < 1.0
    
    @pytest.mark.asyncio
    async def test_inject_constraints(self, engine, sample_context):
        """Test injecting constraints into execution plan."""
        plan = {
            "name": "Test Plan",
            "steps": [
                {"name": "Step 1", "action": "retrieve"},
                {"name": "Step 2", "action": "analyze"}
            ]
        }
        
        constrained_plan = await engine.inject_constraints(plan, sample_context)
        
        assert "domain_constraints" in constrained_plan
        assert constrained_plan["domain_constraints"]["domain_name"] == "test"
        assert "compliance_requirements" in constrained_plan
        
        # Check steps were modified
        for step in constrained_plan["steps"]:
            assert step["validation_required"] is True
            assert step["domain_context"] == "test"
    
    @pytest.mark.asyncio
    async def test_check_compliance(self, engine, sample_context):
        """Test comprehensive compliance checking."""
        execution_result = {
            "actions": [
                {"user_id": "123", "type": "data_access", "has_approval": True},
                {"user_id": "456", "type": "data_access", "has_approval": False}
            ]
        }
        
        report = await engine.check_compliance(execution_result, sample_context)
        
        assert isinstance(report, ComplianceReport)
        assert report.domain_name == "test"
        assert len(report.rule_results) == 2
        assert report.overall_compliance is False  # Second action should fail
        assert report.compliance_score < 1.0
        assert len(report.recommendations) > 0
    
    def test_generate_recommendations(self, engine, sample_context):
        """Test recommendation generation."""
        results = [
            ValidationResult(
                is_valid=False,
                violations=["Policy violation: approval required"],
                compliance_score=0.0
            ),
            ValidationResult(
                is_valid=False,
                violations=["Policy violation: approval required", "Data access violation"],
                compliance_score=0.0
            )
        ]
        
        recommendations = engine._generate_recommendations(results, sample_context)
        
        assert len(recommendations) > 0
        assert any("recurring violation" in rec for rec in recommendations)
        assert any("approval" in rec for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_load_rules(self, engine):
        """Test loading rules for a domain."""
        rules = [
            BusinessRule(
                rule_id="rule_001",
                name="Test Rule",
                description="A test rule",
                rule_type="expression",
                condition="true",
                action="allow"
            )
        ]
        
        await engine.load_rules("finance", rules)
        
        assert "finance" in engine.rule_cache
        assert engine.rule_cache["finance"] == rules
    
    @pytest.mark.asyncio
    async def test_get_applicable_rules(self, engine):
        """Test getting applicable rules for a domain."""
        rules = [
            BusinessRule(
                rule_id="rule_001",
                name="General Rule",
                description="Applies to all actions",
                rule_type="expression",
                condition="true",
                action="allow",
                enabled=True
            ),
            BusinessRule(
                rule_id="rule_002",
                name="Specific Rule",
                description="Applies to specific actions",
                rule_type="expression",
                condition="true",
                action="allow",
                enabled=True,
                metadata={"action_types": ["data_access"]}
            ),
            BusinessRule(
                rule_id="rule_003",
                name="Disabled Rule",
                description="This rule is disabled",
                rule_type="expression",
                condition="true",
                action="allow",
                enabled=False
            )
        ]
        
        await engine.load_rules("finance", rules)
        
        # Test getting all rules
        all_rules = await engine.get_applicable_rules("finance")
        assert len(all_rules) == 2  # Disabled rule should be excluded
        
        # Test getting rules for specific action type
        specific_rules = await engine.get_applicable_rules("finance", "data_access")
        assert len(specific_rules) == 2  # General rule + specific rule
        
        # Test getting rules for non-matching action type
        no_rules = await engine.get_applicable_rules("finance", "other_action")
        assert len(no_rules) == 1  # Only general rule