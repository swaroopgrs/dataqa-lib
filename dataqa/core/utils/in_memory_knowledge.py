import logging
from typing import Dict, Optional

import yaml

from dataqa.core.utils.data_model_util import create_base_model

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Knowledge base object"""

    def __init__(self, config: Dict):
        """
        :param config: config dictionary that defines all retrievable
        """
        self.config = config
        self.data = self.ingest_knowledge_base()

    def get_kb_by_name(self, kb_name: str) -> Optional[Dict]:
        """
        :param kb_name: string of knowledge base name
        :return: knowledge base with given name
        """
        for kb in self.data:
            if kb["name"] == kb_name:
                return kb
        return None

    def get_kb_by_index(self, kb_index: str) -> Optional[Dict]:
        """
        :param kb_index: string of knowledge base index
        :return: knowledge base with given index
        """
        for kb in self.data:
            if kb["knowledge_base_index"] == kb_index:
                return kb
        return None

    def ingest_knowledge_base(self):
        # TODO: validate retrievable data path
        retrievable_data = yaml.safe_load(
            open(self.config["retrievable_data_path"], "r")
        )
        knowledge_base = []
        for retrievable in self.config["data"]:
            name = retrievable["name"]
            fields = retrievable["fields"]
            knowledge_base_index = retrievable["knowledge_base_index"]

            record_base_model = create_base_model(name, fields)

            data = retrievable_data[name]["data"]
            parsed_data_list = []
            for record in data:
                try:
                    parsed_data = record_base_model.model_validate(record)
                    parsed_data_list.append(parsed_data)
                except:
                    logger.error(
                        f"Failed to parse record for {name} retrievable. Record:\n{record}"
                    )

            knowledge_base.append(
                {
                    "name": name,
                    "base_model": record_base_model,
                    "knowledge_base_index": knowledge_base_index,
                    "records": parsed_data_list,
                }
            )
        return knowledge_base


# if __name__ == "__main__":
#     retriever_config = yaml.safe_load(
#         open("example/ccb_risk/config/config_retriever.yml", "r")
#     )
#     my_kb = KnowledgeBase(retriever_config["knowledge_base"])
#     print()

