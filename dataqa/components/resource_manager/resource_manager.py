import logging
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from dataqa.data_models.asset_models import (
    DatabaseSchema,
    Example,
    Examples,
    ResourceType,
    Rule,
    Rules,
    TableSchema,
)
from dataqa.services.storage import BaseDataSource

logger = logging.getLogger(__name__)


class ResourceManagerConfig(BaseModel):
    asset_directory: str


class ResourceManager:
    rules: Optional[Rules] = None
    schema: Optional[DatabaseSchema] = None
    examples: Optional[Examples] = None

    def __init__(self, data_source: BaseDataSource):
        """
        Initialize the resource manager with a data source.
        
        Args:
           data_source: An object that implements the BaseDataSource interface
                        for reading raw asset data.
        """
        self.data_source = data_source
        self.load()

    def load(self):
        """
        Load all resources using the provided data source.
        The parsing logic is here and is NOT duplicated.
        """
        try:
            raw_rules = self.data_source.read_asset("rules.yml")
            self.rules = Rules(**raw_rules)
            logger.info(f"Loaded {len(self.rules.rules)} rules.")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load or parse rules.yml: {e}")

        try:
            raw_schema = self.data_source.read_asset("schema.yml")
            self.schema = DatabaseSchema(**raw_schema)
            logger.info(f"Loaded {len(self.schema.tables)} tables into schema.")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load or parse schema.yml: {e}")

        try:
            raw_examples = self.data_source.read_asset("examples.yml")
            self.examples = Examples(**raw_examples)
            logger.info(f"Loaded {len(self.examples.examples)} examples.")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load or parse examples.yml: {e}")

    def get_resource(
        self, resource_type: ResourceType, module_name: str
    ) -> List[Union[Rule, Example, TableSchema]]:
        """
        Retrieves a list of resources, handling module-specific and global assets.
        """
        if resource_type == ResourceType.Rule:
            if not self.rules:
                return []
            return [
                rule for rule in self.rules.rules
                if rule.module_name == module_name or not rule.module_name
            ]
        elif resource_type == ResourceType.Example:
            if not self.examples:
                return []
            return [
                example for example in self.examples.examples
                if example.module_name == module_name or not example.module_name
            ]
        elif resource_type == ResourceType.Schema:
            if not self.schema:
                return []
            return self.schema.tables
        
        return []