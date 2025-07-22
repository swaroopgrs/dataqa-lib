"""
Tests for domain knowledge management system.
"""

import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.dataqa.orchestration.domain.knowledge import (
    DomainKnowledgeManager,
    KnowledgeSource,
    KnowledgeVersion,
    RuleMapping
)
from src.dataqa.orchestration.models import (
    BusinessRule,
    DomainContext,
    Policy,
    RegulatoryRequirement,
    SchemaConstraint
)


class TestKnowledgeSource:
    """Test KnowledgeSource model."""
    
    def test_knowledge_source_creation(self):
        """Test creating a knowledge source."""
        source = KnowledgeSource(
            name="Test Source",
            source_type="file",
            location="/path/to/file.yaml",
            format="yaml"
        )
        
        assert source.name == "Test Source"
        assert source.source_type == "file"
        assert source.location == "/path/to/file.yaml"
        assert source.format == "yaml"
        assert source.enabled is True
        assert source.refresh_interval_minutes == 60
    
    def test_knowledge_source_defaults(self):
        """Test knowledge source with default values."""
        source = KnowledgeSource(
            name="Test Source",
            source_type="memory",
            location="memory"
        )
        
        assert source.format == "yaml"
        assert source.refresh_interval_minutes == 60
        assert source.enabled is True


class TestRuleMapping:
    """Test RuleMapping model."""
    
    def test_rule_mapping_creation(self):
        """Test creating a rule mapping."""
        mapping = RuleMapping(
            source_rule_id="rule_123",
            target_domain="finance",
            priority=1,
            conditions={"type": "validation"},
            transformations={"field": "value"}
        )
        
        assert mapping.source_rule_id == "rule_123"
        assert mapping.target_domain == "finance"
        assert mapping.priority == 1
        assert mapping.conditions == {"type": "validation"}
        assert mapping.transformations == {"field": "value"}
        assert mapping.enabled is True


class TestKnowledgeVersion:
    """Test KnowledgeVersion model."""
    
    def test_knowledge_version_creation(self):
        """Test creating a knowledge version."""
        version = KnowledgeVersion(
            domain_name="finance",
            version_number="1.0.0",
            created_by="admin",
            description="Initial version",
            changes=["Added basic rules"]
        )
        
        assert version.domain_name == "finance"
        assert version.version_number == "1.0.0"
        assert version.created_by == "admin"
        assert version.description == "Initial version"
        assert version.changes == ["Added basic rules"]
        assert version.is_active is False


class TestDomainKnowledgeManager:
    """Test DomainKnowledgeManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a domain knowledge manager for testing."""
        return DomainKnowledgeManager()
    
    @pytest.fixture
    def sample_knowledge_data(self):
        """Sample knowledge data for testing."""
        return {
            "business_rules": [
                {
                    "rule_id": "rule_001",
                    "name": "Data Access Rule",
                    "description": "Validate data access permissions",
                    "rule_type": "validation",
                    "condition": "has_field('user_id')",
                    "action": "validate_access",
                    "domain": "finance",
                    "enabled": True
                },
                {
                    "rule_id": "rule_002",
                    "name": "Amount Validation",
                    "description": "Validate transaction amounts",
                    "rule_type": "constraint",
                    "condition": "field_greater_than('amount', 0)",
                    "action": "reject_negative",
                    "domain": "finance",
                    "enabled": True
                }
            ],
            "schema_constraints": [
                {
                    "constraint_id": "constraint_001",
                    "name": "Required User ID",
                    "schema_path": "$.user_id",
                    "constraint_type": "required",
                    "constraint_value": True,
                    "error_message": "User ID is required",
                    "applicable_domains": ["finance"]
                }
            ],
            "regulatory_requirements": [
                {
                    "requirement_id": "req_001",
                    "regulation_name": "SOX Compliance",
                    "requirement_text": "All financial transactions must be logged",
                    "compliance_check": "audit_log_exists",
                    "applicable_domains": ["finance"],
                    "severity": "critical"
                }
            ],
            "organizational_policies": [
                {
                    "policy_id": "policy_001",
                    "name": "Data Access Policy",
                    "description": "Controls data access permissions",
                    "policy_text": "Users must have explicit permission for data access",
                    "enforcement_level": "mandatory"
                }
            ]
        }
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert isinstance(manager.knowledge_sources, dict)
        assert isinstance(manager.rule_mappings, dict)
        assert isinstance(manager.knowledge_versions, dict)
        assert isinstance(manager.domain_contexts, dict)
        assert isinstance(manager.cached_knowledge, dict)
    
    def test_configuration_loading_yaml(self, manager, sample_knowledge_data):
        """Test loading configuration from YAML file."""
        config_data = {
            "knowledge_sources": [
                {
                    "source_id": "source_001",
                    "name": "Finance Rules",
                    "source_type": "file",
                    "location": "/path/to/finance.yaml",
                    "format": "yaml"
                }
            ],
            "rule_mappings": [
                {
                    "mapping_id": "mapping_001",
                    "source_rule_id": "rule_001",
                    "target_domain": "finance",
                    "priority": 1
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager._load_configuration(config_path)
            
            assert len(manager.knowledge_sources) == 1
            assert "source_001" in manager.knowledge_sources
            assert manager.knowledge_sources["source_001"].name == "Finance Rules"
            
            assert "finance" in manager.rule_mappings
            assert len(manager.rule_mappings["finance"]) == 1
            assert manager.rule_mappings["finance"][0].source_rule_id == "rule_001"
        finally:
            Path(config_path).unlink()
    
    def test_configuration_loading_json(self, manager):
        """Test loading configuration from JSON file."""
        config_data = {
            "knowledge_sources": [
                {
                    "source_id": "source_002",
                    "name": "HR Rules",
                    "source_type": "memory",
                    "location": "memory",
                    "format": "json"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            manager._load_configuration(config_path)
            
            assert len(manager.knowledge_sources) == 1
            assert "source_002" in manager.knowledge_sources
            assert manager.knowledge_sources["source_002"].name == "HR Rules"
        finally:
            Path(config_path).unlink()
    
    def test_configuration_file_not_found(self, manager):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            manager._load_configuration("/nonexistent/path.yaml")
    
    @pytest.mark.asyncio
    async def test_register_knowledge_source(self, manager):
        """Test registering a knowledge source."""
        source = KnowledgeSource(
            name="Test Source",
            source_type="memory",
            location="memory"
        )
        
        with patch.object(manager, '_refresh_knowledge_source') as mock_refresh:
            await manager.register_knowledge_source(source)
            
            assert source.source_id in manager.knowledge_sources
            assert manager.knowledge_sources[source.source_id] == source
            mock_refresh.assert_called_once_with(source.source_id)
    
    @pytest.mark.asyncio
    async def test_load_from_file_yaml(self, manager, sample_knowledge_data):
        """Test loading knowledge from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_knowledge_data, f)
            file_path = f.name
        
        try:
            source = KnowledgeSource(
                name="Test Source",
                source_type="file",
                location=file_path,
                format="yaml"
            )
            
            await manager._load_from_file(source)
            
            assert source.source_id in manager.cached_knowledge
            cached_data = manager.cached_knowledge[source.source_id]
            assert "business_rules" in cached_data
            assert len(cached_data["business_rules"]) == 2
        finally:
            Path(file_path).unlink()
    
    @pytest.mark.asyncio
    async def test_load_from_file_json(self, manager, sample_knowledge_data):
        """Test loading knowledge from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_knowledge_data, f)
            file_path = f.name
        
        try:
            source = KnowledgeSource(
                name="Test Source",
                source_type="file",
                location=file_path,
                format="json"
            )
            
            await manager._load_from_file(source)
            
            assert source.source_id in manager.cached_knowledge
            cached_data = manager.cached_knowledge[source.source_id]
            assert "business_rules" in cached_data
            assert len(cached_data["business_rules"]) == 2
        finally:
            Path(file_path).unlink()
    
    @pytest.mark.asyncio
    async def test_load_from_file_unsupported_format(self, manager):
        """Test handling of unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write("<root></root>")
            file_path = f.name
        
        try:
            source = KnowledgeSource(
                name="Test Source",
                source_type="file",
                location=file_path,
                format="xml"
            )
            
            with pytest.raises(ValueError, match="Unsupported format: xml"):
                await manager._load_from_file(source)
        finally:
            Path(file_path).unlink()
    
    @pytest.mark.asyncio
    async def test_load_domain_context(self, manager, sample_knowledge_data):
        """Test loading domain context with all components."""
        # Setup cached knowledge
        source_id = "test_source"
        manager.cached_knowledge[source_id] = sample_knowledge_data
        manager.knowledge_sources[source_id] = KnowledgeSource(
            source_id=source_id,
            name="Test Source",
            source_type="memory",
            location="memory"
        )
        
        context = await manager.load_domain_context("finance")
        
        assert context is not None
        assert context.domain_name == "finance"
        assert len(context.applicable_rules) == 2
        assert len(context.schema_constraints) == 1
        assert len(context.regulatory_requirements) == 1
        assert len(context.organizational_policies) == 1
        
        # Verify caching
        assert "finance" in manager.domain_contexts
        assert manager.domain_contexts["finance"] == context
    
    @pytest.mark.asyncio
    async def test_collect_business_rules(self, manager, sample_knowledge_data):
        """Test collecting business rules for a domain."""
        # Setup cached knowledge
        source_id = "test_source"
        manager.cached_knowledge[source_id] = sample_knowledge_data
        manager.knowledge_sources[source_id] = KnowledgeSource(
            source_id=source_id,
            name="Test Source",
            source_type="memory",
            location="memory"
        )
        
        rules = await manager._collect_business_rules("finance")
        
        assert len(rules) == 2
        assert all(isinstance(rule, BusinessRule) for rule in rules)
        assert rules[0].name == "Data Access Rule"
        assert rules[1].name == "Amount Validation"
    
    @pytest.mark.asyncio
    async def test_collect_schema_constraints(self, manager, sample_knowledge_data):
        """Test collecting schema constraints for a domain."""
        # Setup cached knowledge
        source_id = "test_source"
        manager.cached_knowledge[source_id] = sample_knowledge_data
        manager.knowledge_sources[source_id] = KnowledgeSource(
            source_id=source_id,
            name="Test Source",
            source_type="memory",
            location="memory"
        )
        
        constraints = await manager._collect_schema_constraints("finance")
        
        assert len(constraints) == 1
        assert isinstance(constraints[0], SchemaConstraint)
        assert constraints[0].name == "Required User ID"
    
    @pytest.mark.asyncio
    async def test_collect_regulatory_requirements(self, manager, sample_knowledge_data):
        """Test collecting regulatory requirements for a domain."""
        # Setup cached knowledge
        source_id = "test_source"
        manager.cached_knowledge[source_id] = sample_knowledge_data
        manager.knowledge_sources[source_id] = KnowledgeSource(
            source_id=source_id,
            name="Test Source",
            source_type="memory",
            location="memory"
        )
        
        requirements = await manager._collect_regulatory_requirements("finance")
        
        assert len(requirements) == 1
        assert isinstance(requirements[0], RegulatoryRequirement)
        assert requirements[0].regulation_name == "SOX Compliance"
    
    @pytest.mark.asyncio
    async def test_collect_organizational_policies(self, manager, sample_knowledge_data):
        """Test collecting organizational policies for a domain."""
        # Setup cached knowledge
        source_id = "test_source"
        manager.cached_knowledge[source_id] = sample_knowledge_data
        manager.knowledge_sources[source_id] = KnowledgeSource(
            source_id=source_id,
            name="Test Source",
            source_type="memory",
            location="memory"
        )
        
        policies = await manager._collect_organizational_policies("finance")
        
        assert len(policies) == 1
        assert isinstance(policies[0], Policy)
        assert policies[0].name == "Data Access Policy"
    
    def test_rule_applies_to_domain_direct(self, manager):
        """Test rule application through direct domain specification."""
        rule_data = {
            "rule_id": "rule_001",
            "name": "Test Rule",
            "domain": "finance"
        }
        
        assert manager._rule_applies_to_domain(rule_data, "finance", [])
        assert not manager._rule_applies_to_domain(rule_data, "hr", [])
    
    def test_rule_applies_to_domain_mapping(self, manager):
        """Test rule application through rule mapping."""
        rule_data = {
            "rule_id": "rule_001",
            "name": "Test Rule",
            "type": "validation"
        }
        
        mapping = RuleMapping(
            source_rule_id="rule_001",
            target_domain="finance",
            conditions={"type": "validation"}
        )
        
        assert manager._rule_applies_to_domain(rule_data, "finance", [mapping])
        assert not manager._rule_applies_to_domain(rule_data, "hr", [mapping])
    
    def test_constraint_applies_to_domain(self, manager):
        """Test constraint application to domain."""
        constraint_data = {
            "name": "Test Constraint",
            "applicable_domains": ["finance", "hr"]
        }
        
        assert manager._constraint_applies_to_domain(constraint_data, "finance")
        assert manager._constraint_applies_to_domain(constraint_data, "hr")
        assert not manager._constraint_applies_to_domain(constraint_data, "marketing")
        
        # Test constraint with no domain restrictions
        constraint_no_domains = {"name": "Universal Constraint"}
        assert manager._constraint_applies_to_domain(constraint_no_domains, "finance")
    
    def test_evaluate_mapping_conditions(self, manager):
        """Test evaluation of mapping conditions."""
        rule_data = {
            "type": "validation",
            "priority": 1,
            "enabled": True
        }
        
        # Test matching conditions
        conditions = {"type": "validation", "priority": 1}
        assert manager._evaluate_mapping_conditions(rule_data, conditions)
        
        # Test non-matching conditions
        conditions = {"type": "constraint", "priority": 1}
        assert not manager._evaluate_mapping_conditions(rule_data, conditions)
        
        # Test empty conditions (should match)
        assert manager._evaluate_mapping_conditions(rule_data, {})
    
    @pytest.mark.asyncio
    async def test_update_knowledge(self, manager):
        """Test updating domain knowledge."""
        knowledge_data = {
            "business_rules": [
                {
                    "rule_id": "rule_new",
                    "name": "New Rule",
                    "description": "A new business rule",
                    "rule_type": "validation",
                    "condition": "true",
                    "action": "allow"
                }
            ]
        }
        
        version_info = {
            "version_number": "2.0.0",
            "created_by": "admin",
            "description": "Added new rule",
            "changes": ["Added rule_new"]
        }
        
        version_id = await manager.update_knowledge("finance", knowledge_data, version_info)
        
        # Check version was created
        assert "finance" in manager.knowledge_versions
        versions = manager.knowledge_versions["finance"]
        assert len(versions) == 1
        assert versions[0].version_id == version_id
        assert versions[0].is_active
        assert versions[0].version_number == "2.0.0"
        
        # Check knowledge was cached
        memory_source_id = "memory_finance"
        assert memory_source_id in manager.cached_knowledge
        assert manager.cached_knowledge[memory_source_id] == knowledge_data
        
        # Check memory source was created
        assert memory_source_id in manager.knowledge_sources
        
        # Check domain context was invalidated
        manager.domain_contexts["finance"] = DomainContext(domain_name="finance")
        await manager.update_knowledge("finance", knowledge_data)
        assert "finance" not in manager.domain_contexts
    
    @pytest.mark.asyncio
    async def test_get_knowledge_versions(self, manager):
        """Test getting knowledge versions for a domain."""
        # Add some versions
        version1 = KnowledgeVersion(
            domain_name="finance",
            version_number="1.0.0",
            created_by="admin",
            description="Initial version"
        )
        version2 = KnowledgeVersion(
            domain_name="finance",
            version_number="2.0.0",
            created_by="admin",
            description="Updated version"
        )
        
        manager.knowledge_versions["finance"] = [version1, version2]
        
        versions = await manager.get_knowledge_versions("finance")
        assert len(versions) == 2
        assert versions[0] == version1
        assert versions[1] == version2
        
        # Test non-existent domain
        versions = await manager.get_knowledge_versions("nonexistent")
        assert versions == []
    
    @pytest.mark.asyncio
    async def test_get_active_version(self, manager):
        """Test getting active version for a domain."""
        version1 = KnowledgeVersion(
            domain_name="finance",
            version_number="1.0.0",
            created_by="admin",
            description="Initial version",
            is_active=False
        )
        version2 = KnowledgeVersion(
            domain_name="finance",
            version_number="2.0.0",
            created_by="admin",
            description="Updated version",
            is_active=True
        )
        
        manager.knowledge_versions["finance"] = [version1, version2]
        
        active_version = await manager.get_active_version("finance")
        assert active_version == version2
        
        # Test no active version
        version1.is_active = False
        version2.is_active = False
        active_version = await manager.get_active_version("finance")
        assert active_version is None
    
    @pytest.mark.asyncio
    async def test_refresh_all_sources(self, manager):
        """Test refreshing all knowledge sources."""
        source1 = KnowledgeSource(
            source_id="source1",
            name="Source 1",
            source_type="memory",
            location="memory"
        )
        source2 = KnowledgeSource(
            source_id="source2",
            name="Source 2",
            source_type="memory",
            location="memory"
        )
        
        manager.knowledge_sources["source1"] = source1
        manager.knowledge_sources["source2"] = source2
        manager.domain_contexts["finance"] = DomainContext(domain_name="finance")
        
        with patch.object(manager, '_refresh_knowledge_source') as mock_refresh:
            await manager.refresh_all_sources()
            
            assert mock_refresh.call_count == 2
            mock_refresh.assert_any_call("source1")
            mock_refresh.assert_any_call("source2")
            
            # Check domain contexts were cleared
            assert len(manager.domain_contexts) == 0
    
    @pytest.mark.asyncio
    async def test_get_domain_names(self, manager, sample_knowledge_data):
        """Test getting all available domain names."""
        # Setup rule mappings
        manager.rule_mappings["finance"] = []
        manager.rule_mappings["hr"] = []
        
        # Setup cached knowledge
        source_id = "test_source"
        manager.cached_knowledge[source_id] = sample_knowledge_data
        
        domain_names = await manager.get_domain_names()
        
        assert "finance" in domain_names
        assert "hr" in domain_names
        assert len(domain_names) >= 2