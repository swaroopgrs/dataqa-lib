"""
Domain knowledge management for multi-agent orchestration.
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ..models import (
    BusinessRule, 
    DomainContext, 
    Policy, 
    RegulatoryRequirement, 
    SchemaConstraint
)


class KnowledgeSource(BaseModel):
    """Configuration for a knowledge source."""
    source_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    source_type: str  # "file", "database", "api", "memory"
    location: str  # Path, URL, or connection string
    format: str = "yaml"  # "yaml", "json", "xml", "database"
    refresh_interval_minutes: int = 60
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RuleMapping(BaseModel):
    """Mapping configuration for rules to domains."""
    mapping_id: str = Field(default_factory=lambda: str(uuid4()))
    source_rule_id: str
    target_domain: str
    priority: int = 1
    conditions: Dict[str, Any] = Field(default_factory=dict)
    transformations: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class KnowledgeVersion(BaseModel):
    """Version information for knowledge artifacts."""
    version_id: str = Field(default_factory=lambda: str(uuid4()))
    domain_name: str
    version_number: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    description: str
    changes: List[str] = Field(default_factory=list)
    is_active: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DomainKnowledgeManager:
    """
    Manager for domain-specific knowledge and business rules with configurable sources.
    
    Supports multiple knowledge sources, rule mappings, and versioning for evolving
    business rules and compliance requirements.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the domain knowledge manager."""
        self.knowledge_sources: Dict[str, KnowledgeSource] = {}
        self.rule_mappings: Dict[str, List[RuleMapping]] = {}
        self.knowledge_versions: Dict[str, List[KnowledgeVersion]] = {}
        self.domain_contexts: Dict[str, DomainContext] = {}
        self.cached_knowledge: Dict[str, Dict[str, Any]] = {}
        
        if config_path:
            self._load_configuration(config_path)
    
    def _load_configuration(self, config_path: str) -> None:
        """Load knowledge manager configuration from file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Load knowledge sources
        for source_config in config.get('knowledge_sources', []):
            source = KnowledgeSource(**source_config)
            self.knowledge_sources[source.source_id] = source
        
        # Load rule mappings
        for mapping_config in config.get('rule_mappings', []):
            mapping = RuleMapping(**mapping_config)
            domain = mapping.target_domain
            if domain not in self.rule_mappings:
                self.rule_mappings[domain] = []
            self.rule_mappings[domain].append(mapping)
    
    async def register_knowledge_source(self, source: KnowledgeSource) -> None:
        """Register a new knowledge source."""
        self.knowledge_sources[source.source_id] = source
        await self._refresh_knowledge_source(source.source_id)
    
    async def _refresh_knowledge_source(self, source_id: str) -> None:
        """Refresh knowledge from a specific source."""
        source = self.knowledge_sources.get(source_id)
        if not source or not source.enabled:
            return
        
        try:
            if source.source_type == "file":
                await self._load_from_file(source)
            elif source.source_type == "memory":
                # Memory sources are managed directly
                pass
            # Add other source types as needed
        except Exception as e:
            # Log error but don't fail completely
            print(f"Error refreshing knowledge source {source_id}: {e}")
    
    async def _load_from_file(self, source: KnowledgeSource) -> None:
        """Load knowledge from a file source."""
        file_path = Path(source.location)
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                if source.format == "yaml":
                    data = yaml.safe_load(f)
                elif source.format == "json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported format: {source.format}")
            
            # Cache the loaded knowledge
            self.cached_knowledge[source.source_id] = data
        except Exception as e:
            # Log error but don't fail completely
            print(f"Error loading from file {source.location}: {e}")
            if source.format not in ["yaml", "json"]:
                raise ValueError(f"Unsupported format: {source.format}")
    
    async def load_domain_context(self, domain_name: str) -> Optional[DomainContext]:
        """Load domain context by name with all applicable rules and constraints."""
        if domain_name in self.domain_contexts:
            return self.domain_contexts[domain_name]
        
        # Collect rules from all sources
        business_rules = await self._collect_business_rules(domain_name)
        schema_constraints = await self._collect_schema_constraints(domain_name)
        regulatory_requirements = await self._collect_regulatory_requirements(domain_name)
        organizational_policies = await self._collect_organizational_policies(domain_name)
        
        context = DomainContext(
            domain_name=domain_name,
            applicable_rules=business_rules,
            schema_constraints=schema_constraints,
            regulatory_requirements=regulatory_requirements,
            organizational_policies=organizational_policies
        )
        
        self.domain_contexts[domain_name] = context
        return context
    
    async def _collect_business_rules(self, domain_name: str) -> List[BusinessRule]:
        """Collect business rules applicable to a domain."""
        rules = []
        
        # Check rule mappings for this domain
        mappings = self.rule_mappings.get(domain_name, [])
        
        for source_id, source in self.knowledge_sources.items():
            if not source.enabled:
                continue
            
            knowledge = self.cached_knowledge.get(source_id, {})
            source_rules = knowledge.get('business_rules', [])
            
            for rule_data in source_rules:
                # Apply domain filtering
                if self._rule_applies_to_domain(rule_data, domain_name, mappings):
                    rule = BusinessRule(**rule_data)
                    rules.append(rule)
        
        return rules
    
    async def _collect_schema_constraints(self, domain_name: str) -> List[SchemaConstraint]:
        """Collect schema constraints applicable to a domain."""
        constraints = []
        
        for source_id, source in self.knowledge_sources.items():
            if not source.enabled:
                continue
            
            knowledge = self.cached_knowledge.get(source_id, {})
            source_constraints = knowledge.get('schema_constraints', [])
            
            for constraint_data in source_constraints:
                if self._constraint_applies_to_domain(constraint_data, domain_name):
                    constraint = SchemaConstraint(**constraint_data)
                    constraints.append(constraint)
        
        return constraints
    
    async def _collect_regulatory_requirements(self, domain_name: str) -> List[RegulatoryRequirement]:
        """Collect regulatory requirements applicable to a domain."""
        requirements = []
        
        for source_id, source in self.knowledge_sources.items():
            if not source.enabled:
                continue
            
            knowledge = self.cached_knowledge.get(source_id, {})
            source_requirements = knowledge.get('regulatory_requirements', [])
            
            for req_data in source_requirements:
                if domain_name in req_data.get('applicable_domains', []):
                    requirement = RegulatoryRequirement(**req_data)
                    requirements.append(requirement)
        
        return requirements
    
    async def _collect_organizational_policies(self, domain_name: str) -> List[Policy]:
        """Collect organizational policies applicable to a domain."""
        policies = []
        
        for source_id, source in self.knowledge_sources.items():
            if not source.enabled:
                continue
            
            knowledge = self.cached_knowledge.get(source_id, {})
            source_policies = knowledge.get('organizational_policies', [])
            
            for policy_data in source_policies:
                # Policies may apply to all domains or specific ones
                policy = Policy(**policy_data)
                policies.append(policy)
        
        return policies
    
    def _rule_applies_to_domain(self, rule_data: Dict[str, Any], domain_name: str, mappings: List[RuleMapping]) -> bool:
        """Check if a rule applies to a specific domain."""
        # Check direct domain specification
        if rule_data.get('domain') == domain_name:
            return True
        
        # Check rule mappings
        rule_id = rule_data.get('rule_id')
        for mapping in mappings:
            if mapping.source_rule_id == rule_id and mapping.enabled and mapping.target_domain == domain_name:
                # Apply mapping conditions if any
                if self._evaluate_mapping_conditions(rule_data, mapping.conditions):
                    return True
        
        return False
    
    def _constraint_applies_to_domain(self, constraint_data: Dict[str, Any], domain_name: str) -> bool:
        """Check if a constraint applies to a specific domain."""
        # Simple domain matching - can be enhanced with more complex logic
        applicable_domains = constraint_data.get('applicable_domains', [])
        return not applicable_domains or domain_name in applicable_domains
    
    def _evaluate_mapping_conditions(self, rule_data: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """Evaluate mapping conditions for rule applicability."""
        if not conditions:
            return True
        
        # Simple condition evaluation - can be enhanced with expression engine
        for key, expected_value in conditions.items():
            if rule_data.get(key) != expected_value:
                return False
        
        return True
    
    async def update_knowledge(self, domain_name: str, knowledge: Dict[str, Any], version_info: Optional[Dict[str, Any]] = None) -> str:
        """Update domain knowledge and create a new version."""
        # Create version record
        version = KnowledgeVersion(
            domain_name=domain_name,
            version_number=version_info.get('version_number', '1.0.0') if version_info else '1.0.0',
            created_by=version_info.get('created_by', 'system') if version_info else 'system',
            description=version_info.get('description', 'Knowledge update') if version_info else 'Knowledge update',
            changes=version_info.get('changes', []) if version_info else []
        )
        
        # Store version
        if domain_name not in self.knowledge_versions:
            self.knowledge_versions[domain_name] = []
        
        # Deactivate previous versions
        for prev_version in self.knowledge_versions[domain_name]:
            prev_version.is_active = False
        
        version.is_active = True
        self.knowledge_versions[domain_name].append(version)
        
        # Update cached knowledge
        memory_source_id = f"memory_{domain_name}"
        if memory_source_id not in self.knowledge_sources:
            source = KnowledgeSource(
                source_id=memory_source_id,
                name=f"Memory source for {domain_name}",
                source_type="memory",
                location="memory",
                format="dict"
            )
            self.knowledge_sources[memory_source_id] = source
        
        self.cached_knowledge[memory_source_id] = knowledge
        
        # Invalidate cached domain context
        if domain_name in self.domain_contexts:
            del self.domain_contexts[domain_name]
        
        return version.version_id
    
    async def get_knowledge_versions(self, domain_name: str) -> List[KnowledgeVersion]:
        """Get all versions for a domain."""
        return self.knowledge_versions.get(domain_name, [])
    
    async def get_active_version(self, domain_name: str) -> Optional[KnowledgeVersion]:
        """Get the active version for a domain."""
        versions = self.knowledge_versions.get(domain_name, [])
        for version in versions:
            if version.is_active:
                return version
        return None
    
    async def refresh_all_sources(self) -> None:
        """Refresh knowledge from all enabled sources."""
        for source_id in self.knowledge_sources:
            await self._refresh_knowledge_source(source_id)
        
        # Clear cached domain contexts to force reload
        self.domain_contexts.clear()
    
    async def get_domain_names(self) -> Set[str]:
        """Get all available domain names."""
        domains = set()
        
        # From rule mappings
        domains.update(self.rule_mappings.keys())
        
        # From cached knowledge
        for knowledge in self.cached_knowledge.values():
            for rule in knowledge.get('business_rules', []):
                if rule.get('domain'):
                    domains.add(rule['domain'])
        
        return domains