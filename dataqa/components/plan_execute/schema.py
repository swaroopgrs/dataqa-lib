from enum import Enum
from operator import add
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, model_validator

from dataqa.llm.base_llm import LLMOutput


class WorkerName(Enum):
    RetrievalWorker = "retrieval_worker"
    AnalyticsWorker = "analytics_worker"
    PlotWorker = "plot_worker"


class Task(BaseModel):
    """One individual task"""

    worker: WorkerName = Field(
        description="the worker that should be called for solving the task"
    )
    task_description: str = Field(description="the description of the task")


class Plan(BaseModel):
    """The plan that consists a list of tasks."""

    tasks: List[Task] = Field(
        default_factory=list, description="A list of tasks "
    )

    def summarize(self):
        if not self.tasks:
            return "No plan generated yet."
        tasks = []
        for i, task in enumerate(self.tasks):
            tasks.append(
                (
                    f"Step {i + 1}:\n"
                    f"  Worker: {task.worker.value}\n"
                    f"  Task: {task.task_description}\n"
                )
            )
        return "".join(tasks)


class TaskResponse(Task):
    response: str = Field(description="Summarize the execution of one task")


class WorkerResponse(BaseModel):
    """The list of completed tasks and their response"""

    task_response: List[TaskResponse] = Field(default_factory=list)

    def summarize(self):
        if not self.task_response:
            return "No tasks completed yet."
        tasks = []
        for i, task in enumerate(self.task_response):
            tasks.append(
                (
                    f"Completed Task {i + 1}:\n"
                    f"  Worker: {task.worker.value}\n"
                    f"  Task: {task.task_description}\n"
                    f"  Execution response: {task.response}\n"
                )
            )
        return "".join(tasks)


class Response(BaseModel):
    """Response to user. It could contain a text response, some dataframes and some images."""

    response: str = Field(description="Text response to the user.")
    output_df_name: List[str] = Field(
        description="The names of a list of dataframes to be displayed to the user."
    )
    output_img_name: List[str] = Field(
        description="The names of a list of images to displayed to the user."
    )


class Action(Enum):
    Continue = "continue"
    Return = "return"


class PlannerAct(BaseModel):
    """
    Action to perform in the next.

    This model contains three attributes: action, plan and response.
    - If action="continue", then generate a plan.
    - If action="return", then generate a response message.

    Example
    -------
    - If more tasks are required to complete the objective, action="continue" and generate a new plan as a list of tasks and their assigned workers.
    {
        "action": "continue",
        "plan": {
            "tasks": [
                {
                    "worker": "retrieval_worker",
                    "task_description": "a retrieval task"
                },
                {
                    "worker": "analytics_worker",
                    "task_description": "an analytics task"
                }
            ]
        }
    }

    - If prompt back to clarify the question, generate a response message to be returned.
    {
        "action": "return",
        "response": "prompt back message"
    }
    """

    action: Action = Field(
        description="the action to take in the next. Either 'continue' or 'return'"
    )
    plan: Plan = Field(
        description="The plan to follow. Required when action='continue'.",
        default=None,
    )
    response: str = Field(
        description="The prompt back message to be returned. Required when action='return'",
        default=None,
    )

    @model_validator(mode="after")
    def validate_action(self) -> "PlannerAct":
        if self.action == Action.Continue:
            if not isinstance(self.plan, Plan):
                err_msg = (
                    f"Plan is required when action == '{Action.Continue.value}'"
                )
                raise ValueError(err_msg)
        elif self.action == Action.Return:
            if not isinstance(self.response, str):
                err_msg = f"Response is required when action == '{Action.Return.value}'"
                raise ValueError(err_msg)
        else:
            err_msg = f"action should be either {Action.Continue.value} or {Action.Return.value}"
        return self


class ReplannerAct(BaseModel):
    """
    Action to perform in the next.

    This model contains three attributes: action, plan and response.
    - If action="continue", then generate a plan.
    - If action="return", then generate a response.

    Example
    -------
    - If more tasks are required to complete the objective, action="continue" and generate a new plan as a list of tasks and their assigned workers.
    {
        "action": "continue",
        "plan": {
            "tasks": [
                {
                    "worker": "retrieval_worker",
                    "task_description": "a retrieval task"
                },
                {
                    "worker": "analytics_worker",
                    "task_description": "an analytics task"
                }
            ]
        }
    }

    - If no more tasks are needed, generate a response with a text message and a list of dataframes and images to be returned.
    {
        "action": "return",
        "response": {
            "response": "text message",
            "output_df_name": ["df1", "df2"],
            "output_img_name": ["img1", "img2"]
        }
    }
    """

    action: Action = Field(
        description="the action to take in the next. Either 'continue' or 'return'"
    )
    plan: Plan = Field(
        description="The plan to follow. Required when action='continue'.",
        default=None,
    )
    response: Response = Field(
        description="The response to be returned. Required when action='return'",
        default=None,
    )

    @model_validator(mode="after")
    def validate_action(self) -> "ReplannerAct":
        if self.action == Action.Continue:
            if not isinstance(self.plan, Plan):
                err_msg = (
                    f"Plan is required when action == '{Action.Continue.value}'"
                )
                raise ValueError(err_msg)
        elif self.action == Action.Return:
            if not isinstance(self.response, Response):
                err_msg = f"Response is required when action == '{Action.Return.value}'"
                raise ValueError(err_msg)
        else:
            err_msg = f"action should be either {Action.Continue.value} or {Action.Return.value}"
        return self


def worker_response_reducer(
    res1: WorkerResponse, res2: WorkerResponse
) -> WorkerResponse:
    return WorkerResponse(task_response=res1.task_response + res2.task_response)


class PlanExecuteState(BaseModel):
    query: str
    final_response: Optional[Response] = None
    plan: Annotated[List[Plan], add] = Field(default_factory=list)
    worker_response: Annotated[WorkerResponse, worker_response_reducer] = (
        WorkerResponse()
    )
    llm_output: Annotated[List[LLMOutput], add] = Field(
        default_factory=list,
        description="the list of llm calls triggered by planner and replanner",
    )
    history: List[str] = Field(
        default_factory=list,
        description="List of conversation history between cwd agent and user",
    )
