from typing import Callable, Dict

import yaml

from dataqa.services.storage import BaseDataSource


class DBCDataSource(BaseDataSource):
    def __init__(self, s3_callable: Callable, asset_s3_prefix: str):
        """
        A DataSource that reads assets from S3 using a provided callable.

        Args:
            s3_callable: A function with a signature like `s3_callable(s3_path: str, mode: str) -> bytes`.
                         It is only used for reading ('r') in this context.
            asset_s3_prefix: The base S3 prefix where assets like 'rules.yml' are stored.
        """
        self.s3_callable = s3_callable
        self.asset_s3_prefix = asset_s3_prefix

    def read_asset(self, asset_name: str) -> Dict:
        s3_path = f"{self.asset_s3_prefix.rstrip('/')}/{asset_name}"
        # Assumes s3_callable reads and returns the raw byte content of the file
        raw_content = self.s3_callable(s3_path, mode="r")
        return yaml.safe_load(raw_content)
