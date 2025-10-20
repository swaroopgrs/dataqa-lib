import logging
import os
import pickle
import time
import traceback
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from dataqa.benchmark.amap import amap
from dataqa.benchmark.llm_judge_prompt import LLM_JUDGE_PROMPT
from dataqa.benchmark.log import get_logger
from dataqa.benchmark.schema import (
    COMPLETE_EVAL_RESULT,
    COMPLETE_TEST_RESULT,
    TEST_RESULT_DATAFRAME,
    TEST_RESULT_FILE,
    TEST_RESULT_FULL_STATE,
    TEST_RESULT_IMAGE,
    BenchmarkConfig,
    BenchmarkUseCaseConfig,
    EvaluationLabel,
    LLMJudgeOutput,
    Prediction,
    TestDataItem,
    TestResultItem,
    UseCaseTestData,
)
from dataqa.benchmark.utils import out_yaml
from dataqa.core.client import CoreRequest, CoreResponse
from dataqa.core.llm.openai import AzureOpenAI, AzureOpenAIConfig
from dataqa.core.pipelines.pipeline import build_graph_from_config
from dataqa.core.pipelines.schema import PipelineConfig

# from dataqa.core.state import PipelineInput
from dataqa.core.utils.agent_util import (
    dataframe_to_llm_judge_string,
    image_to_llm_judge_string,
)
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    # CONFIGURABLE,
    # DEFAULT_THREAD,
    # THREAD_ID,
    TOKEN,
)
from dataqa.core.utils.prompt_utils import build_prompt
from dataqa.integrations.local.client import LocalClient
from dataqa.scripts.azure_token import get_az_token_using_cert


def convert_enum_to_str(data):
    if isinstance(data, dict):
        return {k: convert_enum_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_enum_to_str(item) for item in data]
    elif isinstance(data, Enum):
        return data.value
    else:
        return data


class TestPipeline:
    """Test pipeline for CWD benchmarking"""

    test_data: List[UseCaseTestData] = []
    test_result: List[List[TestResultItem]] = []

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        self.output = Path(config.output)
        self.output.mkdir(parents=True, exist_ok=True)

        if not config.log:
            config.log = os.path.join(config.output, "test.log")
        Path(config.log).parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(
            name="TestPipeline",
            file_path=config.log,
            level=logging.DEBUG if config.debug else logging.INFO,
        )

        self.logger.info("Init test pipeline")
        self.logger.info(f"Test output saved to {config.output}")
        self.logger.info(f"Test log saved to {config.log}")

        self.load_test_data()

        if self.config.resume:
            self.load_test_result()

        if self.config.run_llm_eval:
            self.llm_judge_model = AzureOpenAI(
                AzureOpenAIConfig(
                    model=self.config.llm_judge_model,
                    api_version="2024-08-01-preview",
                    api_type="azure",
                    temperature=0,
                    with_structured_output=LLMJudgeOutput,
                )
            )
            self.llm_judge_prompt = build_prompt(LLM_JUDGE_PROMPT)

    def load_test_data(self):
        self.logger.info(
            f"Load test data for {len(self.config.use_case_config)} use cases..."
        )

        for config in self.config.use_case_config:
            if not os.path.isfile(config.test_data_file):
                self.logger.warning(
                    f"Test data file {config.test_data_file} does NOT exist. Skip use case {config.name}."
                )
                continue

            self.logger.debug(f"Load test data from {config.test_data_file}...")

            test_id_list = config.test_id_list
            data = yaml.safe_load(open(config.test_data_file))
            data = UseCaseTestData(**data)
            data.data = [x for x in data.data if x.active]
            self.logger.info(
                f"Load {len(data.data)} active test examples for use case {config.name}"
            )
            if test_id_list is not None:
                data.data = [x for x in data.data if x.id in test_id_list]
                self.logger.info(
                    f"Filter {len(data.data)} test examples for use case {config.name}"
                )

            self.test_data.append(data)

        self.logger.info("Loading test data completed.")

    def get_test_result_path(
        self, config: BenchmarkUseCaseConfig, data: TestDataItem
    ):
        return self.output / config.name / f"{data.id}"

    def load_one_test_result(self, path: Path) -> Union[TestResultItem, None]:
        if os.path.isfile(path):
            try:
                with open(path, "r") as file:
                    data = yaml.safe_load(file)
            except Exception:
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        data = yaml.safe_load(file)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load test result from {path}: {e}"
                    )
                    return None
            try:
                test_result_item = TestResultItem(**data)
            except Exception as e:
                self.logger.warning(
                    f"Failed to load test result from {path}: {e}"
                )
                return None
            return test_result_item
        else:
            return None

    def load_test_result(self):
        self.logger.info("Load previous test results...")
        self.test_result = []
        total_num_results = 0
        for config, test_data in zip(
            self.config.use_case_config, self.test_data
        ):
            test_result = []
            for data in test_data.data:
                path = (
                    self.get_test_result_path(config, data) / TEST_RESULT_FILE
                )
                test_result.append(self.load_one_test_result(path))

            self.test_result.append(test_result)

            num_results = len([x for x in test_result if x])
            total_num_results += num_results

            self.logger.info(
                f"Load {num_results} previous test results for use case {config.name}"
            )

        self.logger.info(
            f"Load {total_num_results} previous test results in total."
        )

    def save_dataframe(self, df, name, path):
        df.to_csv(path / f"{name}.csv", encoding="utf-8", index=False)

    def save_image(self, binary, df, name, path):
        with open(path / f"{name}.png", "wb") as f:
            f.write(binary)
        self.save_dataframe(df, name, path)

    def save_test_result(self, test_result: TestResultItem):
        # save the complete test result in yaml
        path = Path(test_result.local_path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / TEST_RESULT_FILE, "w", encoding="utf-8") as f:
            try:
                out_yaml.dump(convert_enum_to_str(test_result.model_dump()), f)
            except Exception:
                self.logger.warning(
                    f"Failed to dump test result: {test_result.model_dump()}"
                )

    def save_raw_prediction(
        self,
        path: Union[str, Path],
        run_id: int,
        response: CoreResponse,
        solution_type: Literal["agent", "pipeline"] = "agent",
    ):
        if isinstance(path, str):
            path = Path(path)
        path = path / str(run_id)
        path.mkdir(parents=True, exist_ok=True)

        # Save the CoreResponse text
        with open(path / TEST_RESULT_FULL_STATE, "w", encoding="utf-8") as f:
            f.write(response.text)

        # Save dataframes
        df_path = path / TEST_RESULT_DATAFRAME
        df_path.mkdir(parents=True, exist_ok=True)
        for i, df in enumerate(response.output_dataframes):
            df_name = f"dataframe_{i + 1}"
            self.save_dataframe(df, df_name, df_path)
        
        # Save images
        img_path = path / TEST_RESULT_IMAGE
        img_path.mkdir(parents=True, exist_ok=True)
        for i, img_bytes in enumerate(response.output_images):
            img_name = f"output_image_{i + 1}.png"
            with open(img_path / img_name, "wb") as f:
                f.write(img_bytes)

    def combine_final_response(
        self, path: Union[str, Path], run_id: int, response: CoreResponse
    ) -> str:
        if not isinstance(response, CoreResponse):
            return "no response"
        if isinstance(path, str):
            path = Path(path)
        run_path = path / str(run_id)
        text = f"{response.text.strip()}\n"

        # Load dataframes
        for i, df in enumerate(response.output_dataframes):
            df_name = f"dataframe_{i + 1}.csv"
            fn = run_path / TEST_RESULT_DATAFRAME / df_name
            if not os.path.isfile(fn):
                self.logger.warning(f"Dataframe {df_name} is not found.")
                continue
            try:
                df = pd.read_csv(fn)
                text += f"\n{dataframe_to_llm_judge_string(df_name, df)}"
            except Exception as e:
                self.logger.warning(f"Failed to load dataframe from {fn} : {e}")
                text += f"\ndataframe: {df_name}\nFailed to load data"

        # Load images
        for i, img_bytes in enumerate(response.output_images):
            img_name = f"output_image_{i + 1}.jpg"
            fn_img = run_path / TEST_RESULT_IMAGE / img_name
            if not os.path.isfile(fn_img):
                self.logger.warning(f"Image {img_name} is not found.")
                continue
            try:
                with open(fn_img, "rb") as f:
                    img_data = f.read()
                text += f"\n{image_to_llm_judge_string(img_name, img_data)}"
                # Assuming image_to_llm_judge_string can handle image data
            except Exception as e:
                self.logger.warning(f"Failed to load image from {fn_img} : {e}")
                text += f"\nimage: {img_name}\nFailed to load data"

        return text

    async def run_prediction_for_one_test_data(
        self, inputs: Tuple[BenchmarkUseCaseConfig, TestDataItem, int, int]
    ):
        config, data, idx, total = inputs

        predictions = []
        local_path = str(self.get_test_result_path(config=config, data=data))

        self.logger.debug(f"Test question ({idx}): {data.question}")
        for run_id in range(self.config.num_run):
            # build agent, start state, LG config
            if self.config.solution_type == "agent":
                client = LocalClient(config_path=str(config.cwd_config))
                
                request = CoreRequest(
                    user_query=data.question,
                    question_id=data.id,
                    conversation_id=f"test_session_{idx}_{run_id}",
                )
                start_time = time.time()
                try:
                    # run the agentic workflow in batch mode
                    response: CoreResponse = None
                    async for chunk in client.process_query(
                        request,
                        streaming=False,
                        summarize=False,
                        prompt_back=False,
                    ):
                        if isinstance(chunk, CoreResponse):
                            response = chunk

                    # parse the final response and summary
                    final_response = response.text
                    if response.output_dataframes:
                        final_response += "\n📊 Output DataFrames:"
                        for i, df in enumerate(response.output_dataframes):
                            final_response += f"\n--- DataFrame {i + 1} ---"
                            # Using to_markdown for clean console output
                            final_response += df.to_markdown(index=False)

                    self.logger.debug(f"final_response: {final_response}")

                    self.logger.debug(
                        f"Test question {{idx}} run {run_id} response: {repr(response.text)}"
                    )

                    summary = ""
                    for step in response.steps:
                        summary += f"\n--- {step.name} ---"
                        summary += step.content
                except Exception as e:
                    summary = (
                        f"LocalClient run failed: {traceback.format_exc()}"
                    )
                    self.logger.warning(
                        f"LocalClient run failed for test example {data.id} use case {config.name}: {repr(e)}"
                    )
                self.save_raw_prediction(
                    path=local_path,
                    run_id=run_id,
                    response=response,
                    solution_type="agent",
                )

                predictions.append(
                    Prediction(
                        run_id=run_id,
                        dataframes=[
                            f"dataframe_{i + 1}"
                            for i in range(len(response.output_dataframes))
                        ],
                        images=[
                            f"output_image_{i + 1}"
                            for i in range(len(response.output_images))
                        ],
                        final_response=final_response,
                        combined_response=self.combine_final_response(
                            path=local_path, run_id=run_id, response=response
                        ),
                        summary=summary,
                        datetime=str(datetime.now()),
                        latency=time.time() - start_time,
                    )
                )
            elif self.config.solution_type == "pipeline":
                base_dir = os.environ.get("BASE_DIR", ".")
                config_path = os.path.join(base_dir, config.cwd_config)
                pipeline_config = yaml.safe_load(
                    open(config_path).read().format(BASE_DIR=base_dir)
                )
                pipeline_config = yaml.safe_load(pipeline_config)
                pipeline_schema = PipelineConfig(**pipeline_config)

                workflow, state_base_model = build_graph_from_config(
                    pipeline_schema=pipeline_schema
                )

                # previous_rewritten_query = ""
                # TODO: FIX pipeline benchmarking
                # state = state_base_model(
                #     input=PipelineInput(
                #         query=data.question,
                #         previous_rewritten_query=previous_rewritten_query,
                #     )
                # )
                # 
                # if os.environ.get("CERT_PATH"):
                #     token = get_az_token_using_cert()
                #     runnable_config = {
                #         CONFIGURABLE: {
                #             THREAD_ID: DEFAULT_THREAD,
                #             API_KEY: os.environ.get("AZURE_OPENAI_API_KEY", ""),
                #             BASE_URL: os.environ.get("AZURE_ENDPOINT", ""),
                #             TOKEN: os.environ.get("AZURE_OPENAI_API_TOKEN", token),
                #         }
                #     }
                # else:
                #     runnable_config = {
                #         CONFIGURABLE: {
                #             THREAD_ID: DEFAULT_THREAD,
                #             API_KEY: os.environ.get("AZURE_OPENAI_API_KEY", ""),
                #             BASE_URL: os.environ.get("AZURE_ENDPOINT", ""),
                #             TOKEN: os.environ.get("AZURE_OPENAI_API_TOKEN", ""),
                #         }
                #     }
                start_time = time.time()

        test_result_item = TestResultItem(
            use_case_config=config,
            local_path=local_path,
            input_data=data,
            predictions=predictions,
        )

        self.save_test_result(test_result=test_result_item)

        if idx % self.config.batch_size == 0:
            self.logger.info(
                f"Complete prediction job ({idx} / {total}) in use case {config.name}."
            )
    

    async def run_prediction_for_one_use_case(
        self,
        config: BenchmarkUseCaseConfig,
        data: UseCaseTestData,
        result: List[TestResultItem],
    ):
        self.logger.info(
            f"Generating predictions for use case {config.name}..."
        )

        tasks = []
        len_test_data = len(data.data)

        for i in range(len_test_data):
            if result[i] is None:
                tasks.append((config, data.data[i], i, len_test_data))

        if not tasks:
            self.logger.info(
                f"No unfinished experiment for use case {config.name}"
            )
            return

        await amap(
            self.run_prediction_for_one_test_data,
            tasks,
            limit=self.config.batch_size,
        )

        self.logger.info(
            f"Finished generating predictions for use case {config.name}."
        )

    async def run_prediction(self):
        self.logger.info("Working on generating predictions...")

        for config, data, result in zip(
            self.config.use_case_config, self.test_data, self.test_result
        ):
            await self.run_prediction_for_one_use_case(config, data, result)

        self.load_test_result()

        fn = self.output / COMPLETE_TEST_RESULT
        with open(fn, "wb") as f:
            pickle.dump(self.test_result, f)

        self.logger.info(f"Finished generating predictions. Saved at {str(fn)}")

    async def run_llm_eval_for_one_test_data(
        self,
        inputs: Tuple[
            BenchmarkUseCaseConfig, TestDataItem, TestResultItem, int, int
        ],
    ):
        config, data, test_result, idx, total = inputs

        instruction = data.instruction_for_llm_judge
        if instruction:
            instruction = f"Follow the instructions below in your evaluation:\n{instruction.strip()}\n"
        
        if test_result is None:
            self.logger.warning(
                f"Skip LLM-judg evaluation for test example {data.id}."
            )
            return
        
        # from dataqa.integrations.local.generate_token_multi import get_access_token
        # new_token = get_access_token()
        # new_token = get_az_token_using_cert()[0]

        for prediction in test_result.predictions:
            if prediction.evaluation.llm_label != EvaluationLabel.NotAvailable:
                # has already been evaluated
                continue
            if not data.ground_truth_output:
                # no ground truth
                prediction.evaluation.llm_judge_output = LLMJudgeOutput(
                    REASON="no ground truth", SCORE="0"
                )
                prediction.evaluation.llm_label = EvaluationLabel.NotAvailable

            elif prediction.final_response is None:
                # no test result
                prediction.evaluation.llm_judge_output = LLMJudgeOutput(
                    REASON="no final response generated", SCORE=0
                )
                prediction.evaluation.llm_label = EvaluationLabel.Wrong

            else:
                if os.environ.get("CERT_PATH"):
                    # print(f"Initializing LLM using CERT_PATH: {os.environ.get('CERT_PATH')}")
                    token = ""
                    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
                    if api_key == "":
                        print(
                            "Running Standard LLM Azure API Subscription....."
                        )
                        api_key = get_az_token_using_cert()[0]
                    else:
                        print(
                            "Running Multi-Tenant LLM Azure API Subscription....."
                        )
                        token = get_az_token_using_cert()[0]
                    api_args = {
                        API_KEY: api_key,
                        BASE_URL: os.environ.get("AZURE_ENDPOINT", ""),
                        TOKEN: token,
                    }
                else:
                    api_args = {
                        # For local mode, we assume credentials are in env vars
                        API_KEY: os.environ.get("AZURE_OPENAI_API_KEY", ""),
                        BASE_URL: os.environ.get("AZURE_ENDPOINT", ""),
                        TOKEN: os.environ.get("AZURE_OPENAI_API_TOKEN", ""),
                    }
                llm_judge_output = await self.llm_judge_model.ainvoke(
                    messages=self.llm_judge_prompt.invoke(
                        dict(
                            question=data.question.strip(),
                            ground_truth_response=data.ground_truth_output.strip(),
                            instruction=instruction,
                            prediction=prediction.combined_response,
                        )
                    ),
                    **api_args,
                )
                if isinstance(llm_judge_output.generation, LLMJudgeOutput):
                    prediction.evaluation.llm_judge_output = (
                        llm_judge_output.generation
                    )
                    if prediction.evaluation.llm_judge_output.SCORE == 1:
                        prediction.evaluation.llm_label = (
                            EvaluationLabel.Correct
                        )
                    elif prediction.evaluation.llm_judge_output.SCORE == -1:
                        prediction.evaluation.llm_label = (
                            EvaluationLabel.PromptBack
                        )
                    elif prediction.evaluation.llm_judge_output.SCORE == -2:
                        prediction.evaluation.llm_label = EvaluationLabel.Reject
                    else:
                        prediction.evaluation.llm_label = EvaluationLabel.Wrong
                else:
                    # parsing error
                    prediction.evaluation.llm_judge_output = LLMJudgeOutput(
                        REASON=f"LLM judge failed: {str(llm_judge_output)}",
                        SCORE=0,
                    )
                    prediction.evaluation.llm_label = (
                        EvaluationLabel.NotAvailable
                    )
            self.logger.debug(
                f"LLM evaluation ({data.id}) run ({prediction.run_id}): {str(prediction.evaluation.llm_judge_output)}"
            )

        self.save_test_result(test_result=test_result)

        if idx % self.config.batch_size == 0:
            self.logger.info(
                f"Complete evaluation job ({idx} / {total}) in use case {config.name}."
            )

    async def run_llm_eval_for_one_use_case(
        self,
        config: BenchmarkUseCaseConfig,
        data: UseCaseTestData,
        result: List[TestResultItem],
    ):
        self.logger.info(f"Running LLM-judge for use case {config.name}...")

        tasks = []
        len_test_data = len(data.data)

        for i in range(len_test_data):
            tasks.append((config, data.data[i], result[i], i, len_test_data))

        await amap(
            self.run_llm_eval_for_one_test_data,
            tasks,
            limit=self.config.batch_size,
        )

        self.logger.info(
            f"Finished LLM-juedge evaluations for use case {config.name}"
        )

    async def run_llm_eval(self):
        self.logger.info("Working on LLM-judge evaluation...")

        for config, data, result in zip(
            self.config.use_case_config, self.test_data, self.test_result
        ):
            await self.run_llm_eval_for_one_use_case(config, data, result)

        self.load_test_result()

        # TODO calculate metric and save results
        fn = self.output / COMPLETE_TEST_RESULT
        with open(fn, "wb") as f:
            pickle.dump(self.test_result, f)

        self.logger.info(f"Finished LLM-judge evaluation. Saved at {str(fn)}")

    def average(
        self,
        metric: Dict[str, List[float]],
        func: Callable = lambda x: float(np.mean(x)),
    ) -> Dict[str, float]:
        result = {}
        total = []
        for name, vals in metric.items():
            result[name] = func(vals)
            total += vals
        result["macro"] = float(np.mean(list(result.values())))
        result["micro"] = func(total)
        return result

    def calculate_matric(self):
        accuracy = {}
        majority_frequency = {}
        latency = {}
        reject_rate = {}
        prompt_back_rate = {}
        prompt_back_example = []
        reject_example = []
        for config, results in zip(
            self.config.use_case_config, self.test_result
        ):
            _correct, _majorify_frequency, _latency, _reject, _prmopt_back = (
                [],
                [],
                [],
                [],
                [],
            )
            for result in results:
                if result is None:
                    continue
                else:
                    _latency += [
                        prediction.latency for prediction in result.predictions
                    ]
                    labels = [
                        prediction.evaluation.llm_label
                        for prediction in result.predictions
                        if prediction.evaluation.llm_label
                        != EvaluationLabel.NotAvailable
                    ]
                if not labels:
                    continue
                count = Counter(labels)
                total = len(labels)

                reject_count = count.get(EvaluationLabel.Reject, 0)
                _reject.append(reject_count)

                prompt_back_count = count.get(EvaluationLabel.PromptBack, 0)
                _prmopt_back.append(prompt_back_count)

                for prediction in result.predictions:
                    if (
                        prediction.evaluation.llm_label
                        == EvaluationLabel.PromptBack
                    ):
                        prompt_back_example.append(
                            (
                                config.name,
                                result.input_data.id,
                                result.input_data.question,
                                prediction.final_response,
                            )
                        )
                    if (
                        prediction.evaluation.llm_label
                        == EvaluationLabel.Reject
                    ):
                        reject_example.append(
                            (
                                config.name,
                                result.input_data.id,
                                result.input_data.question,
                                prediction.final_response,
                            )
                        )

                correct_count = count.get(EvaluationLabel.Correct, 0)
                _correct.append(correct_count)

                if count.get(EvaluationLabel.Wrong, 0):
                    self.logger.debug(
                        f"Failed test question: use case {config.name} question {result.input_data.id} ({count.get(EvaluationLabel.Wrong, 0)}/{total})"
                    )
                _majorify_frequency.append(max(count.values()) / total)

            accuracy[config.name] = [
                (_c, self.config.num_run - _r - _p)
                for _c, _r, _p in zip(_correct, _reject, _prmopt_back)
            ]
            majority_frequency[config.name] = _majorify_frequency
            latency[config.name] = _latency
            reject_rate[config.name] = _reject
            prompt_back_rate[config.name] = _prmopt_back

        self.logger.debug(
            f"Found {len(prompt_back_example)} prompt back responses."
        )
        for usecase, question_id, question, response in prompt_back_example:
            self.logger.debug(f"{usecase} {question_id}")
            self.logger.debug(question)
            self.logger.debug(response)
        self.logger.debug(f"Found {len(reject_example)} reject responses.")
        for usecase, question_id, question, response in reject_example:
            self.logger.debug(f"{usecase} {question_id}")
            self.logger.debug(question)
            self.logger.debug(response)

        accuracy = self.average(
            accuracy, func=lambda x: sum(y[0] for y in x) / sum(y[1] for y in x)
        )
        reject_rate = self.average(
            reject_rate, func=lambda x: sum(x) / len(x) / self.config.num_run
        )
        prompt_back_rate = self.average(
            prompt_back_rate,
            func=lambda x: sum(x) / len(x) / self.config.num_run,
        )
        majority_frequency = self.average(majority_frequency)
        p50 = self.average(latency, func=lambda x: float(np.percentile(x, 50)))
        p90 = self.average(latency, func=lambda x: float(np.percentile(x, 90)))
        p99 = self.average(latency, func=lambda x: float(np.percentile(x, 99)))

        self.logger.info(f"Average accuracy: {accuracy['micro']}")

        metrics = dict(
            accuracy=accuracy,
            reject_rate=reject_rate,
            prompt_back_rate=prompt_back_rate,
            majority_frequency=majority_frequency,
            p50_latency=p50,
            p90_latency=p90,
            p99_latency=p99,
        )

        fn = Path(self.config.output) / COMPLETE_EVAL_RESULT
        with open(fn, "w") as f:
            yaml.dump(metrics, f)
        self.logger.info(f"Overall evaluation results saved at {(str(fn))}")

    async def run(self):
        self.logger.info("Start running experiment...")

        if self.config.run_prediction:
            await self.run_prediction()

        if self.config.run_llm_eval:
            await self.run_llm_eval()

        self.calculate_matric()

        self.logger.info("Experiment has been completed.")

