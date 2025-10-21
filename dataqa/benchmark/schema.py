from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from dataqa.core.components.plan_execute.schema import Response

TEST_RESULT_FILE = "test_result.yml"
TEST_RESULT_FULL_STATE = "full_state.pkl"
TEST_RESULT_DATAFRAME = "dataframe"
TEST_RESULT_IMAGE = "image"
COMPLETE_TEST_RESULT = "complete_test_result.pkl"
COMPLETE_EVAL_RESULT = "evaluation.yml"


class BenchmarkUseCaseConfig(BaseModel):
    name: str
    cwd_config: str  # path to the config on how to build the pipeline
    test_data_file: str  # path to test data
    test_id_list: Optional[List[str]] = None


class BenchmarkConfig(BaseModel):
    use_case_config: List[BenchmarkUseCaseConfig] = Field(default_factory=list)
    output: str
    log: str = ""
    run_prediction: bool = True
    run_llm_eval: bool = True
    llm_judge_model: str = "gpt-4o-2024-08-06"
    batch_size: int = 4
    num_run: int = 1
    run_id: int = 0
    resume: bool = False
    datetime: str = str(datetime.now())  # 2025-05-14 23:57:08.208543
    debug: bool = False
    solution_type: Literal["agent", "pipeline"] = "agent"


class Solution(BaseModel):
    worker: str = ""
    function_name: str = ""
    function_arguments: Any = Field(default_factory=list)


class ComponentGroundTruth(BaseModel):
    worker: str = ""
    component: str = ""
    groundtruth: Any = Field(default_factory=list)


class TestDataItem(BaseModel):
    id: str
    question: str
    active: bool = True
    date_created: str = ""
    previous_question_id: str = ""
    solution: List[Solution] = Field(default_factory=list)
    ground_truth_output: str = None
    component_groundtruth: List[ComponentGroundTruth] = Field(
        default_factory=list
    )
    instruction_for_llm_judge: str = ""
    human_validated: bool = True
    labels: List[str] = Field(default_factory=list)


class UseCaseTestMetadata(BaseModel):
    use_case: str
    as_of_date: str = ""
    schema_file: str = ""
    data_file: Union[str, List[str]] = ""


class UseCaseTestData(BaseModel):
    metadata: UseCaseTestMetadata
    data: List[TestDataItem]


class EvaluationLabel(Enum):
    Correct = "correct"
    Wrong = "wrong"
    NotAvailable = "not available"
    Reject = "reject"
    PromptBack = "prompt back"


class LLMJudgeOutput(BaseModel):
    """Evaluation result of one test example"""

    REASON: str = Field(
        description="The reasoning of how to evaluate the generated answer"
    )
    SCORE: int = Field(
        description="binary score: 1 means the prediction is correct; 0 means the prediction is wrong"
    )


class EvaluationItem(BaseModel):
    human_label: EvaluationLabel = EvaluationLabel.NotAvailable
    llm_label: EvaluationLabel = EvaluationLabel.NotAvailable
    llm_judge_output: Union[LLMJudgeOutput, None] = None


class Prediction(BaseModel):
    run_id: int = 0
    final_response: Union[Response, None, str] = None
    evaluation: EvaluationItem = EvaluationItem()
    combined_response: str = ""
    summary: str = ""
    dataframes: List[str] = Field(
        default_factory=list, description="dataframe names"
    )
    images: List[str] = Field(default_factory=list, description="image names")
    datetime: str = ""
    latency: float = 0


class TestResultItem(BaseModel):
    use_case_config: Union[BenchmarkUseCaseConfig, None] = None
    local_path: str = ""
    input_data: Union[TestDataItem, None] = None
    predictions: List[Prediction] = Field(default_factory=list)
