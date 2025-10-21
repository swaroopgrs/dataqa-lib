from typing import Optional


class PipelineConfigError(Exception):
    def __init__(self, message: Optional[str] = None):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return str(self)
