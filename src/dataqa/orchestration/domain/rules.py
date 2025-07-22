"""
Business rules engine for domain-specific validation and compliance.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ..models import BusinessRule, DomainContext, Policy, RegulatoryRequirement, SchemaConstraint


class ValidationResult(BaseModel):
    """Result of business rule validation."""
    is_valid: bool
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    applied_rules: List[str] = Field(default_factory=list)
    compliance_score: float = 1.0


class ComplianceReport(BaseModel):
    """Comprehensive compliance report."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    domain_name: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    overall_compliance: bool
    compliance_score: float
    rule_results: List[ValidationResult] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RuleValidator(ABC):
    """Abstract base class for rule validators."""
    
    @abstractmethod
    async def validate(self, action: Dict[str, Any], rule: BusinessRule, context: DomainContext) -> ValidationResult:
        """Validate an action against a specific rule."""
        pass
    
    @abstractmethod
    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if this validator supports a specific rule type."""
        pass


class ExpressionRuleValidator(RuleValidator):
    """Validator for expression-based rules."""
    
    def supports_rule_type(self, rule_type: str) -> bool:
        """Support expression and condition rule types."""
        return rule_type in ["expression", "condition", "constraint"]
    
    async def validate(self, action: Dict[str, Any], rule: BusinessRule, context: DomainContext) -> ValidationResult:
        """Validate using expression evaluation."""
        try:
            # Simple expression evaluation - can be enhanced with safer evaluation
            result = self._evaluate_expression(rule.condition, action, context)
            
            if result:
                return ValidationResult(
                    is_valid=True,
                    applied_rules=[rule.rule_id],
                    metadata={"rule_type": rule.rule_type, "expression": rule.condition}
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    violations=[f"Rule violation: {rule.name} - {rule.description}"],
                    applied_rules=[rule.rule_id],
                    compliance_score=0.0,
                    metadata={"rule_type": rule.rule_type, "expression": rule.condition}
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                violations=[f"Rule evaluation error: {rule.name} - {str(e)}"],
                applied_rules=[rule.rule_id],
                compliance_score=0.0,
                metadata={"rule_type": rule.rule_type, "error": str(e)}
            )
    
    def _evaluate_expression(self, expression: str, action: Dict[str, Any], context: DomainContext) -> bool:
        """Evaluate a rule expression safely."""
        # Simple pattern matching for common expressions
        # In production, use a proper expression engine
        
        # Check for field existence
        if "has_field" in expression:
            field_match = re.search(r"has_field\('([^']+)'\)", expression)
            if field_match:
                field_name = field_match.group(1)
                return field_name in action
        
        # Check for field values
        if "field_equals" in expression:
            equals_match = re.search(r"field_equals\('([^']+)',\s*'([^']+)'\)", expression)
            if equals_match:
                field_name, expected_value = equals_match.groups()
                return action.get(field_name) == expected_value
        
        # Check for numeric comparisons
        if "field_greater_than" in expression:
            gt_match = re.search(r"field_greater_than\('([^']+)',\s*(\d+(?:\.\d+)?)\)", expression)
            if gt_match:
                field_name, threshold = gt_match.groups()
                field_value = action.get(field_name)
                if isinstance(field_value, (int, float)):
                    return field_value > float(threshold)
        
        # Default to True for unknown expressions (safe default)
        return True


class PolicyRuleValidator(RuleValidator):
    """Validator for organizational policy rules."""
    
    def supports_rule_type(self, rule_type: str) -> bool:
        """Support policy rule types."""
        return rule_type in ["policy", "organizational_policy"]
    
    async def validate(self, action: Dict[str, Any], rule: BusinessRule, context: DomainContext) -> ValidationResult:
        """Validate against organizational policies."""
        # Find matching policy in context
        matching_policy = None
        for policy in context.organizational_policies:
            if policy.name == rule.name or policy.policy_id == rule.rule_id:
                matching_policy = policy
                break
        
        if not matching_policy:
            return ValidationResult(
                is_valid=True,
                warnings=[f"Policy not found: {rule.name}"],
                applied_rules=[rule.rule_id]
            )
        
        # Check enforcement level
        if matching_policy.enforcement_level == "optional":
            return ValidationResult(
                is_valid=True,
                applied_rules=[rule.rule_id],
                metadata={"enforcement_level": "optional"}
            )
        
        # Apply policy validation logic
        is_valid = self._validate_policy_compliance(action, matching_policy)
        
        if is_valid:
            return ValidationResult(
                is_valid=True,
                applied_rules=[rule.rule_id],
                metadata={"enforcement_level": matching_policy.enforcement_level}
            )
        else:
            # Handle policy violations based on enforcement level
            if matching_policy.enforcement_level == "mandatory":
                return ValidationResult(
                    is_valid=False,
                    violations=[f"Policy violation: {matching_policy.name}"],
                    applied_rules=[rule.rule_id],
                    compliance_score=0.0,
                    metadata={"enforcement_level": matching_policy.enforcement_level}
                )
            elif matching_policy.enforcement_level == "recommended":
                return ValidationResult(
                    is_valid=True,  # Recommended policies don't fail validation
                    warnings=[f"Policy recommendation: {matching_policy.name}"],
                    applied_rules=[rule.rule_id],
                    compliance_score=0.5,
                    metadata={"enforcement_level": matching_policy.enforcement_level}
                )
            else:  # optional
                return ValidationResult(
                    is_valid=True,
                    applied_rules=[rule.rule_id],
                    compliance_score=1.0,
                    metadata={"enforcement_level": matching_policy.enforcement_level}
                )
    
    def _validate_policy_compliance(self, action: Dict[str, Any], policy: Policy) -> bool:
        """Validate compliance with a specific policy."""
        # Simple policy validation - can be enhanced with more sophisticated logic
        action_type = action.get("type", "")
        
        # Check if action type is in exceptions
        if action_type in policy.exceptions:
            return True
        
        # Apply basic policy checks based on policy text
        policy_text_lower = policy.policy_text.lower()
        
        if "approval_required" in policy_text_lower or "approval" in policy_text_lower:
            return action.get("has_approval", False)
        
        if "permission" in policy_text_lower and "data access" in policy_text_lower:
            return action.get("has_approval", False)
        
        if "data_access" in policy_text_lower:
            return action.get("data_access_level", "none") in ["read", "limited"]
        
        # Default to compliant
        return True


class ConstraintValidator:
    """Validator for schema constraints."""
    
    async def validate_constraints(self, action: Dict[str, Any], constraints: List[SchemaConstraint]) -> ValidationResult:
        """Validate action against schema constraints."""
        violations = []
        warnings = []
        applied_constraints = []
        
        for constraint in constraints:
            applied_constraints.append(constraint.constraint_id)
            
            try:
                is_valid = self._validate_constraint(action, constraint)
                if not is_valid:
                    if constraint.severity == "error":
                        violations.append(constraint.error_message)
                    elif constraint.severity == "warning":
                        warnings.append(constraint.error_message)
            except Exception as e:
                violations.append(f"Constraint validation error: {constraint.name} - {str(e)}")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            applied_rules=applied_constraints,
            compliance_score=1.0 if len(violations) == 0 else max(0.0, 1.0 - len(violations) / len(constraints))
        )
    
    def _validate_constraint(self, action: Dict[str, Any], constraint: SchemaConstraint) -> bool:
        """Validate a single constraint."""
        # Extract value using schema path (simplified JSONPath)
        value = self._extract_value(action, constraint.schema_path)
        
        if constraint.constraint_type == "required":
            return value is not None
        elif constraint.constraint_type == "type":
            expected_type = constraint.constraint_value
            if expected_type == "string":
                return isinstance(value, str)
            elif expected_type == "number":
                return isinstance(value, (int, float))
            elif expected_type == "boolean":
                return isinstance(value, bool)
            elif expected_type == "array":
                return isinstance(value, list)
            elif expected_type == "object":
                return isinstance(value, dict)
        elif constraint.constraint_type == "range":
            if isinstance(value, (int, float)) and isinstance(constraint.constraint_value, dict):
                min_val = constraint.constraint_value.get("min")
                max_val = constraint.constraint_value.get("max")
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
                return True
        elif constraint.constraint_type == "pattern":
            if isinstance(value, str):
                pattern = constraint.constraint_value
                return bool(re.match(pattern, value))
        
        return True
    
    def _extract_value(self, data: Dict[str, Any], path: str) -> Any:
        """Extract value from data using a simple path notation."""
        if not path or path == "$":
            return data
        
        # Simple dot notation support
        parts = path.replace("$.", "").split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current


class BusinessRulesEngine:
    """
    Engine for applying business rules and domain constraints with pluggable rule sets.
    
    Supports multiple rule validators, constraint validation, and comprehensive
    compliance reporting.
    """
    
    def __init__(self):
        """Initialize the business rules engine."""
        self.rule_validators: List[RuleValidator] = []
        self.constraint_validator = ConstraintValidator()
        self.rule_cache: Dict[str, List[BusinessRule]] = {}
        
        # Register default validators
        self._register_default_validators()
    
    def _register_default_validators(self) -> None:
        """Register default rule validators."""
        self.rule_validators.extend([
            ExpressionRuleValidator(),
            PolicyRuleValidator()
        ])
    
    def register_validator(self, validator: RuleValidator) -> None:
        """Register a custom rule validator."""
        self.rule_validators.append(validator)
    
    async def validate_action(self, action: Dict[str, Any], context: DomainContext) -> ValidationResult:
        """Validate an action against all applicable business rules."""
        all_violations = []
        all_warnings = []
        all_applied_rules = []
        total_score = 0.0
        rule_count = 0
        
        # Validate business rules
        for rule in context.applicable_rules:
            if not rule.enabled:
                continue
            
            validator = self._find_validator(rule.rule_type)
            if validator:
                result = await validator.validate(action, rule, context)
                all_violations.extend(result.violations)
                all_warnings.extend(result.warnings)
                all_applied_rules.extend(result.applied_rules)
                total_score += result.compliance_score
                rule_count += 1
        
        # Validate schema constraints
        if context.schema_constraints:
            constraint_result = await self.constraint_validator.validate_constraints(
                action, context.schema_constraints
            )
            all_violations.extend(constraint_result.violations)
            all_warnings.extend(constraint_result.warnings)
            all_applied_rules.extend(constraint_result.applied_rules)
            total_score += constraint_result.compliance_score
            rule_count += 1
        
        # Calculate overall compliance score
        overall_score = total_score / rule_count if rule_count > 0 else 1.0
        
        return ValidationResult(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
            warnings=all_warnings,
            applied_rules=all_applied_rules,
            compliance_score=overall_score
        )
    
    def _find_validator(self, rule_type: str) -> Optional[RuleValidator]:
        """Find a validator that supports the given rule type."""
        for validator in self.rule_validators:
            if validator.supports_rule_type(rule_type):
                return validator
        return None
    
    async def inject_constraints(self, plan: Dict[str, Any], context: DomainContext) -> Dict[str, Any]:
        """Inject domain constraints into an execution plan."""
        constrained_plan = plan.copy()
        
        # Add constraint metadata
        constrained_plan["domain_constraints"] = {
            "domain_name": context.domain_name,
            "business_rules_count": len(context.applicable_rules),
            "schema_constraints_count": len(context.schema_constraints),
            "regulatory_requirements_count": len(context.regulatory_requirements),
            "organizational_policies_count": len(context.organizational_policies)
        }
        
        # Add validation checkpoints
        if "steps" in constrained_plan:
            for step in constrained_plan["steps"]:
                step["validation_required"] = True
                step["domain_context"] = context.domain_name
        
        # Add compliance requirements
        constrained_plan["compliance_requirements"] = [
            {
                "requirement_id": req.requirement_id,
                "regulation_name": req.regulation_name,
                "severity": req.severity
            }
            for req in context.regulatory_requirements
        ]
        
        return constrained_plan
    
    async def check_compliance(self, execution_result: Dict[str, Any], context: DomainContext) -> ComplianceReport:
        """Generate comprehensive compliance report for execution results."""
        report = ComplianceReport(
            domain_name=context.domain_name,
            overall_compliance=True,
            compliance_score=1.0
        )
        
        # Validate each action in the execution result
        actions = execution_result.get("actions", [execution_result])
        total_score = 0.0
        
        for action in actions:
            validation_result = await self.validate_action(action, context)
            report.rule_results.append(validation_result)
            total_score += validation_result.compliance_score
            
            if not validation_result.is_valid:
                report.overall_compliance = False
        
        # Calculate overall compliance score
        report.compliance_score = total_score / len(actions) if actions else 1.0
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report.rule_results, context)
        
        return report
    
    def _generate_recommendations(self, results: List[ValidationResult], context: DomainContext) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyze common violation patterns
        violation_patterns = {}
        for result in results:
            for violation in result.violations:
                violation_patterns[violation] = violation_patterns.get(violation, 0) + 1
        
        # Generate recommendations for frequent violations
        for violation, count in violation_patterns.items():
            if count > 1:
                recommendations.append(f"Address recurring violation: {violation} (occurred {count} times)")
        
        # Check for missing approvals
        approval_violations = [v for v in violation_patterns.keys() if "approval" in v.lower() or "policy violation" in v.lower()]
        if approval_violations:
            recommendations.append("Consider implementing automated approval workflows for sensitive operations")
        
        # Check for data access violations
        data_violations = [v for v in violation_patterns.keys() if "data" in v.lower()]
        if data_violations:
            recommendations.append("Review data access policies and implement stricter access controls")
        
        return recommendations
    
    async def load_rules(self, domain_name: str, rules: List[BusinessRule]) -> None:
        """Load business rules for a domain."""
        self.rule_cache[domain_name] = rules
    
    async def get_applicable_rules(self, domain_name: str, action_type: Optional[str] = None) -> List[BusinessRule]:
        """Get applicable rules for a domain and optional action type."""
        domain_rules = self.rule_cache.get(domain_name, [])
        
        # Filter out disabled rules first
        enabled_rules = [rule for rule in domain_rules if rule.enabled]
        
        if not action_type:
            return enabled_rules
        
        # Filter by action type if specified
        applicable_rules = []
        for rule in enabled_rules:
            if not rule.metadata.get("action_types") or action_type in rule.metadata.get("action_types", []):
                applicable_rules.append(rule)
        
        return applicable_rules