"""
Domain context management and utilities with schema-driven rules and entity filtering.
"""

import json
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ..models import DomainContext, BusinessRule, SchemaConstraint, RegulatoryRequirement, Policy


class EntityFilter(BaseModel):
    """Filter configuration for domain entities."""
    filter_id: str = Field(default_factory=lambda: str(uuid4()))
    entity_type: str
    filter_expression: str
    include_fields: Optional[List[str]] = None
    exclude_fields: Optional[List[str]] = None
    conditions: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class SchemaRule(BaseModel):
    """Schema-driven rule definition."""
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    schema_path: str
    rule_type: str  # "validation", "transformation", "filtering"
    rule_expression: str
    target_entities: List[str] = Field(default_factory=list)
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextInjectionConfig(BaseModel):
    """Configuration for domain context injection."""
    config_id: str = Field(default_factory=lambda: str(uuid4()))
    domain_name: str
    injection_points: List[str] = Field(default_factory=list)  # Where to inject context
    entity_filters: List[EntityFilter] = Field(default_factory=list)
    schema_rules: List[SchemaRule] = Field(default_factory=list)
    auto_inject: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DomainContextInjector:
    """Handles injection of domain context into agent workflows."""
    
    def __init__(self):
        """Initialize the context injector."""
        self.injection_configs: Dict[str, ContextInjectionConfig] = {}
        self.entity_schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_injection_config(self, config: ContextInjectionConfig) -> None:
        """Register a context injection configuration."""
        self.injection_configs[config.domain_name] = config
    
    def register_entity_schema(self, entity_type: str, schema: Dict[str, Any]) -> None:
        """Register an entity schema for validation and filtering."""
        self.entity_schemas[entity_type] = schema
    
    async def inject_context(self, workflow_data: Dict[str, Any], domain_name: str, context: DomainContext) -> Dict[str, Any]:
        """Inject domain context into workflow data."""
        config = self.injection_configs.get(domain_name)
        if not config or not config.auto_inject:
            return workflow_data
        
        injected_data = workflow_data.copy()
        
        # Apply entity filters
        injected_data = await self._apply_entity_filters(injected_data, config.entity_filters)
        
        # Apply schema rules
        injected_data = await self._apply_schema_rules(injected_data, config.schema_rules)
        
        # Inject context metadata
        injected_data["domain_context"] = {
            "domain_name": domain_name,
            "rules_count": len(context.applicable_rules),
            "constraints_count": len(context.schema_constraints),
            "policies_count": len(context.organizational_policies),
            "injection_config_id": config.config_id
        }
        
        return injected_data
    
    async def _apply_entity_filters(self, data: Dict[str, Any], filters: List[EntityFilter]) -> Dict[str, Any]:
        """Apply entity filters to workflow data."""
        filtered_data = data.copy()
        
        for entity_filter in filters:
            if not entity_filter.enabled:
                continue
            
            # Find entities of the specified type
            entities = self._find_entities(filtered_data, entity_filter.entity_type)
            
            # Apply filtering
            filtered_entities = []
            for entity in entities:
                if self._entity_matches_filter(entity, entity_filter):
                    filtered_entity = self._apply_field_filters(entity, entity_filter)
                    filtered_entities.append(filtered_entity)
            
            # Update data with filtered entities
            self._update_entities(filtered_data, entity_filter.entity_type, filtered_entities)
        
        return filtered_data
    
    def _find_entities(self, data: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
        """Find entities of a specific type in the data."""
        entities = []
        
        # Simple entity discovery - can be enhanced with more sophisticated logic
        if entity_type in data:
            entity_data = data[entity_type]
            if isinstance(entity_data, list):
                entities.extend(entity_data)
            elif isinstance(entity_data, dict):
                entities.append(entity_data)
        
        # Look for entities in nested structures
        for key, value in data.items():
            if isinstance(value, dict) and value.get("type") == entity_type:
                entities.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("type") == entity_type:
                        entities.append(item)
        
        return entities
    
    def _entity_matches_filter(self, entity: Dict[str, Any], entity_filter: EntityFilter) -> bool:
        """Check if an entity matches the filter conditions."""
        # Evaluate filter expression
        if entity_filter.filter_expression:
            try:
                # Simple expression evaluation - enhance with proper expression engine
                if self._evaluate_filter_expression(entity, entity_filter.filter_expression):
                    return True
            except Exception:
                # If expression evaluation fails, fall back to condition matching
                pass
        
        # Check conditions
        for condition_key, condition_value in entity_filter.conditions.items():
            entity_value = entity.get(condition_key)
            if entity_value != condition_value:
                return False
        
        return True
    
    def _evaluate_filter_expression(self, entity: Dict[str, Any], expression: str) -> bool:
        """Evaluate a filter expression against an entity."""
        # Simple expression evaluation - replace with proper expression engine
        # For now, support basic comparisons
        
        if "==" in expression:
            parts = expression.split("==")
            if len(parts) == 2:
                field_name = parts[0].strip()
                expected_value = parts[1].strip().strip("'\"")
                return str(entity.get(field_name)) == expected_value
        
        if "!=" in expression:
            parts = expression.split("!=")
            if len(parts) == 2:
                field_name = parts[0].strip()
                expected_value = parts[1].strip().strip("'\"")
                return str(entity.get(field_name)) != expected_value
        
        if "contains" in expression:
            # Format: field_name contains 'value'
            parts = expression.split("contains")
            if len(parts) == 2:
                field_name = parts[0].strip()
                search_value = parts[1].strip().strip("'\"")
                field_value = entity.get(field_name, "")
                return search_value in str(field_value)
        
        return True  # Default to include if expression can't be evaluated
    
    def _apply_field_filters(self, entity: Dict[str, Any], entity_filter: EntityFilter) -> Dict[str, Any]:
        """Apply field inclusion/exclusion filters to an entity."""
        filtered_entity = entity.copy()
        
        # Apply field exclusions
        if entity_filter.exclude_fields:
            for field in entity_filter.exclude_fields:
                filtered_entity.pop(field, None)
        
        # Apply field inclusions (if specified, only include these fields)
        if entity_filter.include_fields:
            included_entity = {}
            for field in entity_filter.include_fields:
                if field in filtered_entity:
                    included_entity[field] = filtered_entity[field]
            filtered_entity = included_entity
        
        return filtered_entity
    
    def _update_entities(self, data: Dict[str, Any], entity_type: str, filtered_entities: List[Dict[str, Any]]) -> None:
        """Update the data with filtered entities."""
        if entity_type in data:
            data[entity_type] = filtered_entities
        
        # Update nested structures
        for key, value in data.items():
            if isinstance(value, list):
                updated_list = []
                for item in value:
                    if isinstance(item, dict) and item.get("type") == entity_type:
                        # Find matching filtered entity
                        item_id = item.get("id")
                        matching_entity = next(
                            (e for e in filtered_entities if e.get("id") == item_id),
                            None
                        )
                        if matching_entity:
                            updated_list.append(matching_entity)
                    else:
                        updated_list.append(item)
                data[key] = updated_list
    
    async def _apply_schema_rules(self, data: Dict[str, Any], rules: List[SchemaRule]) -> Dict[str, Any]:
        """Apply schema-driven rules to workflow data."""
        processed_data = data.copy()
        
        # Sort rules by priority
        sorted_rules = sorted(rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            try:
                if rule.rule_type == "validation":
                    # Validation rules don't modify data, just validate
                    self._validate_schema_rule(processed_data, rule)
                elif rule.rule_type == "transformation":
                    processed_data = self._apply_transformation_rule(processed_data, rule)
                elif rule.rule_type == "filtering":
                    processed_data = self._apply_filtering_rule(processed_data, rule)
            except Exception as e:
                # Log error but continue processing
                print(f"Error applying schema rule {rule.rule_id}: {e}")
        
        return processed_data
    
    def _validate_schema_rule(self, data: Dict[str, Any], rule: SchemaRule) -> None:
        """Validate data against a schema rule."""
        # Extract value using schema path
        value = self._extract_value_by_path(data, rule.schema_path)
        
        # Apply validation rule
        if not self._evaluate_rule_expression(value, rule.rule_expression):
            raise ValueError(f"Schema validation failed for rule {rule.rule_id}: {rule.rule_expression}")
    
    def _apply_transformation_rule(self, data: Dict[str, Any], rule: SchemaRule) -> Dict[str, Any]:
        """Apply a transformation rule to data."""
        # Simple transformation - can be enhanced with more sophisticated logic
        transformed_data = data.copy()
        
        # Extract current value
        current_value = self._extract_value_by_path(transformed_data, rule.schema_path)
        
        # Apply transformation based on rule expression
        if rule.rule_expression.startswith("uppercase"):
            if isinstance(current_value, str):
                self._set_value_by_path(transformed_data, rule.schema_path, current_value.upper())
        elif rule.rule_expression.startswith("lowercase"):
            if isinstance(current_value, str):
                self._set_value_by_path(transformed_data, rule.schema_path, current_value.lower())
        elif rule.rule_expression.startswith("default"):
            # Set default value if current value is None or empty
            if not current_value:
                default_value = rule.rule_expression.split("default:")[1].strip()
                self._set_value_by_path(transformed_data, rule.schema_path, default_value)
        
        return transformed_data
    
    def _apply_filtering_rule(self, data: Dict[str, Any], rule: SchemaRule) -> Dict[str, Any]:
        """Apply a filtering rule to data."""
        filtered_data = data.copy()
        
        # Simple filtering based on rule expression
        if rule.rule_expression == "remove_empty":
            value = self._extract_value_by_path(filtered_data, rule.schema_path)
            if not value:
                self._remove_value_by_path(filtered_data, rule.schema_path)
        elif rule.rule_expression.startswith("remove_if"):
            condition = rule.rule_expression.split("remove_if:")[1].strip()
            value = self._extract_value_by_path(filtered_data, rule.schema_path)
            if self._evaluate_rule_expression(value, condition):
                self._remove_value_by_path(filtered_data, rule.schema_path)
        
        return filtered_data
    
    def _extract_value_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """Extract value from data using a path notation."""
        if not path or path == "$":
            return data
        
        # Simple dot notation support
        parts = path.replace("$.", "").split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None
        
        return current
    
    def _set_value_by_path(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set value in data using a path notation."""
        if not path or path == "$":
            return
        
        parts = path.replace("$.", "").split(".")
        current = data
        
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            else:
                return
        
        # Set the final value
        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = value
    
    def _remove_value_by_path(self, data: Dict[str, Any], path: str) -> None:
        """Remove value from data using a path notation."""
        if not path or path == "$":
            return
        
        parts = path.replace("$.", "").split(".")
        current = data
        
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return
        
        # Remove the final value
        final_key = parts[-1]
        if isinstance(current, dict) and final_key in current:
            del current[final_key]
    
    def _evaluate_rule_expression(self, value: Any, expression: str) -> bool:
        """Evaluate a rule expression against a value."""
        # Simple expression evaluation - enhance with proper expression engine
        if expression == "not_empty":
            return value is not None and value != ""
        elif expression == "is_string":
            return isinstance(value, str)
        elif expression == "is_number":
            return isinstance(value, (int, float))
        elif expression.startswith("equals:"):
            expected = expression.split("equals:")[1].strip()
            return str(value) == expected
        elif expression.startswith("contains:"):
            search_term = expression.split("contains:")[1].strip()
            return search_term in str(value)
        
        return True  # Default to True for unknown expressions


class DomainContextManager:
    """
    Manager for domain context creation and management with schema-driven rules and entity filtering.
    
    Provides comprehensive context injection capabilities with configurable entity filtering
    and schema-driven rule application.
    """
    
    def __init__(self):
        """Initialize the domain context manager."""
        self.contexts: Dict[str, DomainContext] = {}
        self.injector = DomainContextInjector()
        self.context_cache: Dict[str, Dict[str, Any]] = {}
    
    def create_context(self, domain_name: str, **kwargs) -> DomainContext:
        """Create a new domain context."""
        context = DomainContext(domain_name=domain_name, **kwargs)
        self.contexts[domain_name] = context
        return context
    
    def get_context(self, domain_name: str) -> Optional[DomainContext]:
        """Get domain context by name."""
        return self.contexts.get(domain_name)
    
    def register_injection_config(self, config: ContextInjectionConfig) -> None:
        """Register a context injection configuration."""
        self.injector.register_injection_config(config)
    
    def register_entity_schema(self, entity_type: str, schema: Dict[str, Any]) -> None:
        """Register an entity schema for validation and filtering."""
        self.injector.register_entity_schema(entity_type, schema)
    
    async def inject_context_into_workflow(self, workflow_data: Dict[str, Any], domain_name: str) -> Dict[str, Any]:
        """Inject domain context into workflow data."""
        context = self.get_context(domain_name)
        if not context:
            return workflow_data
        
        return await self.injector.inject_context(workflow_data, domain_name, context)
    
    def create_entity_filter(self, entity_type: str, filter_expression: str, **kwargs) -> EntityFilter:
        """Create an entity filter configuration."""
        return EntityFilter(
            entity_type=entity_type,
            filter_expression=filter_expression,
            **kwargs
        )
    
    def create_schema_rule(self, schema_path: str, rule_type: str, rule_expression: str, **kwargs) -> SchemaRule:
        """Create a schema rule configuration."""
        return SchemaRule(
            schema_path=schema_path,
            rule_type=rule_type,
            rule_expression=rule_expression,
            **kwargs
        )
    
    def create_injection_config(self, domain_name: str, entity_filters: Optional[List[EntityFilter]] = None, schema_rules: Optional[List[SchemaRule]] = None, **kwargs) -> ContextInjectionConfig:
        """Create a context injection configuration."""
        return ContextInjectionConfig(
            domain_name=domain_name,
            entity_filters=entity_filters or [],
            schema_rules=schema_rules or [],
            **kwargs
        )
    
    async def validate_context_integrity(self, domain_name: str) -> Dict[str, Any]:
        """Validate the integrity of a domain context."""
        context = self.get_context(domain_name)
        if not context:
            return {"valid": False, "error": "Context not found"}
        
        validation_result = {
            "valid": True,
            "domain_name": domain_name,
            "rules_count": len(context.applicable_rules),
            "constraints_count": len(context.schema_constraints),
            "policies_count": len(context.organizational_policies),
            "requirements_count": len(context.regulatory_requirements),
            "issues": []
        }
        
        # Check for rule conflicts
        rule_names = [rule.name for rule in context.applicable_rules]
        duplicate_names = [name for name in rule_names if rule_names.count(name) > 1]
        if duplicate_names:
            validation_result["issues"].append(f"Duplicate rule names: {duplicate_names}")
        
        # Check for constraint conflicts
        constraint_paths = [constraint.schema_path for constraint in context.schema_constraints]
        duplicate_paths = [path for path in constraint_paths if constraint_paths.count(path) > 1]
        if duplicate_paths:
            validation_result["issues"].append(f"Duplicate constraint paths: {duplicate_paths}")
        
        # Check for policy conflicts
        mandatory_policies = [p for p in context.organizational_policies if p.enforcement_level == "mandatory"]
        optional_policies = [p for p in context.organizational_policies if p.enforcement_level == "optional"]
        
        for mandatory in mandatory_policies:
            for optional in optional_policies:
                if mandatory.name == optional.name:
                    validation_result["issues"].append(f"Policy enforcement conflict: {mandatory.name}")
        
        validation_result["valid"] = len(validation_result["issues"]) == 0
        return validation_result