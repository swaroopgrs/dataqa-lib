import os
from abc import ABC, abstractmethod
from typing import Dict

import yaml


class BaseDataSource(ABC):
    @abstractmethod
    def read_asset(self, asset_name: str) -> Dict:
        """Reads a structured asset (like rules.yml) and returns its raw dictionary content."""
        raise NotImplementedError


class LocalFileDataSource(BaseDataSource):
    def __init__(self, asset_directory: str):
        self.asset_directory = asset_directory

    def read_asset(self, asset_name: str) -> Dict:
        """
        Reads a YAML asset file from the local filesystem.

        Args:
            asset_name: The name of the file, e.g., "rules.yml".

        Returns:
            The parsed dictionary content of the YAML file.
        """
        file_path = os.path.join(self.asset_directory, asset_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Asset file not found at: {file_path}")

        with open(file_path, "r") as f:
            return yaml.safe_load(f)
