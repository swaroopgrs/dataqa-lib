from typing import Optional
from pydantic import BaseModel, Field

from dataqa.core.utils.agent_util import NodeName


agent_error_log = []


class Error(BaseModel):
    message: str
    source: str
    cause: Optional[str]
    fix_suggestion: Optional[str]
    # code: Optional[int]  # TODO: define error code for all error messages that will be consumed by CWD agent

    def message_to_agent(self) -> str:
        msg = self.message
        if self.cause is not None:
            msg += f"\nCause: {self.cause}"
        if self.fix_suggestion is not None:
            msg += f"\nSuggestion: {self.fix_suggestion}"
        # agent_error_log.append(msg)
        return msg


InternalDataframeError = Error(
    message="Internal data set is referenced in the SQL, but cannot be used by remote database.",
    source=f"{NodeName.retrieval_worker}.{NodeName.sql_executor}",
    cause="When the internal data set is referenced in the SQL, need to replace it with a subquery that can be used by remote database. No such subquery can be generated.",
    fix_suggestion="Please regenerate the SQL with a subquery that can be used by remote database by using tables available in the remote database and defined in the schema.",
)


class SqlExecutionError(Error):
    message: str = Field(default="Failed to execute SQL on remote database.")
    source: str = Field(
        default=f"{NodeName.retrieval_worker}.{NodeName.sql_executor}"
    )
    fix_suggestion: str = Field(
        default="Please regenerate the SQL and fix the SQL execution error. If needed, please adjust the plan and try different way to solve the problem."
    )


ColumnNamingError = Error(
    message="SQL is executed successfully on remote database, but the column name in the output table is not unique.",
    source=f"{NodeName.retrieval_worker}.{NodeName.sql_generator}",
    cause="If column names in the output table are not unique, it will cause downstream to_json operation failed.",
    fix_suggestion="Please regenerate the SQL and name the columns using alias so that the column names in the output table are unique.",
)
