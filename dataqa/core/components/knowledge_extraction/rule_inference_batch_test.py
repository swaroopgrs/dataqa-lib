import asyncio
import logging
import os
import pickle
from typing import List

import pandas as pd
import yaml
from benchmark.llm_judge_prompt import LLM_JUDGE_PROMPT
from benchmark.log import get_logger
from benchmark.schema import (
    EvaluationLabel,
    LLMJudgeOutput,
    TestDataItem,
    UseCaseTestData,
)
from state import CWDState

from dataqa.agent.cwd_agent.cwd_agent import CWDAgent
from dataqa.components.knowledge_extraction.rule_inference import (
    RuleConsolidation,
    RuleInference,
    RuleTriggered,
    rule_consolidation_prompt_template,
    rule_inference_prompt_template,
    rule_list_str,
    rule_pruning_prompt_template,
)
from dataqa.llm.openai import AzureOpenAI, AzureOpenAIConfig
from dataqa.memory import Memory
from dataqa.utils.agent_util import (
    dataframe_to_llm_judge_string,
    image_to_llm_judge_string,
)
from dataqa.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    DEFAULT_THREAD,
    THREAD_ID,
)
from dataqa.utils.prompt_utils import build_prompt
from dataqa.utils.utils import (
    generate_alphabetic_bullets,
    string_list_to_prompt,
)
from scripts.azure_token import get_az_token_using_cert


class RuleInferenceExperiment:
    def __init__(
        self,
        config_path: str,
        original_config_file: str,
        test_data_file: str,
        output_file_path: str,
        logging_level=logging.INFO,
        max_iteration: int = 3,
    ):
        self.config_path = config_path
        self.original_config_file = original_config_file
        self.test_data_file = test_data_file
        self.test_data = None
        self.output_file_path = output_file_path
        self.max_iteration = max_iteration
        self.logger = get_logger(
            name="RuleInferenceExperiment",
            file_path=f"{output_file_path}.log",
            level=logging_level,
        )
        self.experiment_result = []
        self.consolidated_rules = None
        self.question_id_to_alphabetic_bullets = None

    def get_llm_and_run_config(self):
        api_key = get_az_token_using_cert()[0]
        base_url = os.environ["OPENAI_API_BASE"]
        config = {
            "configurable": {
                "api_key": api_key,
                "base_url": base_url,
            }
        }

        # TODO: move llm config to experiment config file
        llm_config = {
            "model": "gpt-4o-2024-08-06",
            "api_version": "2024-08-01-preview",
            "api_type": "azure_ad",
            "temperature": 0,
            "num_response": 1,
            "azure_model_params": {"model_name": "gpt-4o"},
        }
        llm = AzureOpenAI(**llm_config)
        return llm, config

    async def run_question(self, question: str, custom_instruction: str = None):
        """
        Runs a question through the CWD agent.

        Args:
            question (str): The question to be asked.
            custom_instruction (str, optional): Custom instruction to be included in the agent's prompt.

        Returns:
            Tuple[Optional[str], Optional[str]]:
                - The generated response from the agent.
                - The generated SQL query.

        """
        config_path = self.config_path  # "examples/cib_mp/agent/"
        original_config_file = self.original_config_file
        if custom_instruction is not None:
            config_file = f"{config_path}{original_config_file}"
            agent_config = yaml.safe_load(open(config_file))
            agent_config["prompts"]["use_case_sql_instruction"] = (
                custom_instruction
            )
            agent_config["prompts"]["use_case_planner_instruction"] += (
                f"\n{custom_instruction}"
            )
            updated_config_file = f"{config_path}cwd_agent_prompt_template_custom_instruction.yaml"
            with open(updated_config_file, "w") as f:
                yaml.safe_dump(agent_config, f)
            config_file = updated_config_file
        else:
            config_file = f"{config_path}{original_config_file}"

        agent: CWDAgent = CWDAgent.from_config_path(config_file, Memory())
        state = CWDState(query=question)
        runnable_config = {
            CONFIGURABLE: {
                THREAD_ID: DEFAULT_THREAD,
                API_KEY: get_az_token_using_cert()[0],
                BASE_URL: os.environ["OPENAI_API_BASE"],
            }
        }
        try:
            response, events = await agent(state=state, config=runnable_config)

            text = response.final_response.response
            sql = response.retrieval_worker_state[0].sql_generator_output.sql

            for name in response.final_response.output_df_name:
                df = agent.memory.get_dataframe(name, runnable_config)
                text += f"\n{dataframe_to_llm_judge_string(name, df)}"

            for name in response.final_response.output_img_name:
                df = agent.memory.get_image_data(name, runnable_config)
                text += f"\n{image_to_llm_judge_string(name, df)}"

        except Exception as e:
            self.logger.info(
                f"CWD Agent run failed for test question {question}: {repr(e)}"
            )
            text = None
            sql = None
        return text, sql

    async def llm_eval(self, test_record: TestDataItem, generated_answer: str):
        """
        Evaluate the generated answer using the OpenAI model.

        Args:
            test_record (TestDataItem): The test data record.
            generated_answer (str): The generated answer.

        Returns:
            EvaluationLabel: The label indicating the correctness of the generated answer.

        """
        # TODO: move llm judge config to experiment config file
        llm_judge_model = AzureOpenAI(
            AzureOpenAIConfig(
                model="gpt-4o-2024-08-06",
                api_version="2024-08-01-preview",
                api_type="azure",
                temperature=0,
                with_structured_output=LLMJudgeOutput,
            )
        )
        llm_judge_prompt = build_prompt(LLM_JUDGE_PROMPT)
        instruction = test_record.instruction_for_llm_judge
        if instruction:
            instruction = f"Follow the instructions below in your evaluation:\n{instruction.strip()}\n"

        if not test_record.ground_truth_output:
            # no ground truth
            llm_label = EvaluationLabel.NotAvailable
        else:
            llm_judge_output = await llm_judge_model.ainvoke(
                messages=llm_judge_prompt.invoke(
                    dict(
                        question=test_record.question,
                        ground_truth_response=test_record.ground_truth_output.strip(),
                        instruction=instruction,
                        prediction=generated_answer,
                    )
                ),
                **{
                    API_KEY: get_az_token_using_cert()[0],
                    BASE_URL: os.environ["OPENAI_API_BASE"],
                },
            )
            if isinstance(llm_judge_output.generation, LLMJudgeOutput):
                if llm_judge_output.generation.SCORE == 1:
                    llm_label = EvaluationLabel.Correct
                elif llm_judge_output.generation.SCORE == -1:
                    llm_label = EvaluationLabel.Reject
                else:
                    llm_label = EvaluationLabel.Wrong
            else:
                # parsing error
                llm_label = EvaluationLabel.NotAvailable
        return llm_label

    def load_test_data(self, filter_id: List[str] = None):
        data = yaml.safe_load(open(self.test_data_file))
        data = UseCaseTestData(**data)
        if filter_id is None:
            data.data = [x for x in data.data if x.active]
        else:
            data.data = [x for x in data.data if x.active and x.id in filter_id]
        self.test_data = data

    async def tune_question(self, test_record: TestDataItem):
        """
        Run a test record and tune the rule prompt.

        Args:
            test_record (TestDataItem): The test data item.

        Returns:
            List: The result of the test.

        The result contains the following:

        - The question.
        - The number of iterations.
        - The label of the LLM evaluation.
        - The prompt of the extracted rules.
        - The expected SQL.
        - The generated SQL.
        - The original generated SQL.
        - The ground truth output.
        - The answer.

        """
        expected_sql = test_record.solution[0].function_arguments["sql"]
        self.logger.info(f"Question: {test_record.question}")
        self.logger.info(f"Ground truth SQL:\n{expected_sql}")
        self.logger.info(
            f"Ground truth output:\n{test_record.ground_truth_output}"
        )

        answer, sql = await self.run_question(test_record.question)
        sql_0 = sql
        self.logger.info(f"Answer: {answer}")
        self.logger.info(f"Generated SQL: \n{sql}")

        llm_label = await self.llm_eval(test_record, answer)
        self.logger.info(f"LLM judge: {llm_label}")

        iteration_count = 0
        rule_prompt = ""
        rules = None
        while (
            (llm_label == EvaluationLabel.Wrong)
            or (llm_label == EvaluationLabel.Reject)
        ) and (iteration_count < self.max_iteration):
            iteration_count += 1
            self.logger.info(f"***Iteration: {iteration_count}***")
            llm, config = self.get_llm_and_run_config()
            rule_inference = RuleInference(
                llm=llm, prompt=rule_inference_prompt_template
            )
            rules = await rule_inference(
                query=test_record.question,
                generated_sql=sql_0,
                expected_sql=expected_sql,
                config=config,
            )
            rule_prompt = ""
            for rule in rules["rules"][0].rules:
                rule_prompt += f"- {rule}\n"
            self.logger.info(f"Extracted rules: \n{rule_prompt}")
            answer, sql = await self.run_question(
                test_record.question, rule_prompt
            )
            self.logger.info(f"Answer: {answer}")
            self.logger.info(f"Generated SQL: \n{sql}")
            llm_label = await self.llm_eval(test_record, answer)
            self.logger.info(f"LLM judge: {llm_label}")
        result = [
            test_record.question,
            iteration_count,
            llm_label.value,
            rule_prompt,
            expected_sql,
            sql,
            sql_0,
            test_record.ground_truth_output,
            answer,
        ]
        self.experiment_result.append(result + [test_record.id, rules])
        return result

    async def tune_question_batch(self):
        """
        Runs the `tune_question` function on each test item in the given `test_data` and stores the results in a pandas DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The results are stored in a pickle file and an Excel file.
        """
        result = []
        count = 1
        for item in self.test_data.data:
            self.logger.info(
                f"\n*** {count} of {len(self.test_data.data)} ***\n"
            )
            count += 1
            tune_result = await self.tune_question(item)
            result.append(tune_result)
            pickle.dump(
                self.experiment_result,
                open(f"{self.output_file_path}.pkl", "wb"),
            )
        column_names = [
            "question",
            "iteration_count",
            "llm_label",
            "rule_prompt",
            "expected_sql",
            "generated_sql",
            "generated_sql_0",
            "ground_truth_output",
            "generated_answer",
        ]
        df_result = pd.DataFrame(result, columns=column_names)
        df_result.to_excel(f"{self.output_file_path}.xlsx")

    def prepare_rules_to_combine(self):
        self.logger.info("Preparing rules to combine...")
        question_with_rules = []

        for item in self.experiment_result:
            if item[-1] is not None:
                question_with_rules.append(item)

        alphabetic_bullets = generate_alphabetic_bullets(
            len(question_with_rules)
        )

        question_id_to_alphabetic_bullets = {}
        all_rules, all_prefix = [], []
        for i, item in enumerate(question_with_rules):
            bullet = alphabetic_bullets[i]
            question_id_to_alphabetic_bullets[item[-2]] = bullet
            rules = item[-1]["rules"][0].rules
            prefix = [f"{bullet}{i} - " for i in range(len(rules))]
            all_rules += rules
            all_prefix += prefix
            self.logger.info(f"Question {bullet}: {item[0]}")
            self.logger.info(f"Extracted rules: \n{rules}")
        combined_rules = string_list_to_prompt(all_rules, all_prefix)
        self.question_id_to_alphabetic_bullets = (
            question_id_to_alphabetic_bullets
        )
        self.logger.info(f"Combined rules: \n{combined_rules}")
        return combined_rules

    async def consolidate_rules(self):
        rule_list_str = self.prepare_rules_to_combine()
        llm, config = self.get_llm_and_run_config()
        rule_consolidation = RuleConsolidation(
            llm=llm, prompt=rule_consolidation_prompt_template
        )
        rules = await rule_consolidation(
            rule_list_str=rule_list_str, config=config
        )
        self.consolidated_rules = rules
        rules_list = [rule.rule for rule in rules["rules"][0].rules]
        rules_prompt = string_list_to_prompt(rules_list, "- ")
        self.logger.info(f"Consolidated rules: \n{rules_prompt}")
        return rules_prompt

    async def identify_triggered_rules(self, test_record: TestDataItem):
        llm, config = self.get_llm_and_run_config()
        triggered_rules = RuleTriggered(
            llm=llm, prompt=rule_pruning_prompt_template
        )
        rules = await triggered_rules(
            rule_list_str=rule_list_str,
            query=test_record.question,
            expected_sql=test_record.solution[0].function_arguments["sql"],
            config=config,
        )
        return rules

    async def rule_pruning(self):
        result = []
        triggered_rule_indices = []
        count = 1
        for item in self.test_data.data:
            self.logger.info(
                f"\n*** {count} of {len(self.test_data.data)} ***\n"
            )
            count += 1
            rules_triggered = await self.identify_triggered_rules(item)
            result.append(
                [item.id, item.question, rules_triggered["rules"][0].rules]
            )
            triggered_rule_indices.extend(rules_triggered["rules"][0].rules)
            self.logger.info(f"Result: {result}")
        column_names = ["ID", "question", "triggered_rules"]
        df_result = pd.DataFrame(result, columns=column_names)
        df_result.to_excel(f"{self.output_file_path}.xlsx")
        triggered_rule_indices = sorted(triggered_rule_indices)
        self.logger.info(
            f"Triggered rules list ({len(triggered_rule_indices)}): {triggered_rule_indices}"
        )
        self.logger.info(
            f"Triggered rules set ({len(set(triggered_rule_indices))}): {set(triggered_rule_indices)}"
        )


if __name__ == "__main__":
    os.environ["CERT_PATH"] = ""
    os.environ["CLIENT_ID"] = ""
    os.environ["TENANT_ID"] = ""
    os.environ["OPENAI_API_BASE"] = ""

    # results = pickle.load(open("temp/rule_inference_experiment_cib_mp_20250620_3.pkl", "rb"))
    # print(results)
    # test_data_file = "examples/cib_mp/examples.yaml"
    # test_data = load_test_data(test_data_file)
    # item = test_data.data[16]
    # asyncio.run(tune_question(item))

    train_id_gb = [
        "cib_gb_002",
        "cib_gb_005",
        "cib_gb_006",
        "cib_gb_007",
        "cib_gb_008",
        "cib_gb_009",
        "cib_gb_011",
        "cib_gb_012",
        "cib_gb_014",
        "cib_gb_015",
        "cib_gb_016",
        "cib_gb_017",
        "cib_gb_019",
        "cib_gb_020",
        "cib_gb_021",
        "cib_gb_023",
        "cib_gb_027",
        "cib_gb_028",
        "cib_gb_029",
        "cib_gb_031",
        "cib_gb_032",
        "cib_gb_033",
        "cib_gb_034",
        "cib_gb_035",
        "cib_gb_036",
        "cib_gb_038",
        "cib_gb_039",
        "cib_gb_041",
        "cib_gb_042",
        "cib_gb_043",
        "cib_gb_044",
        "cib_gb_045",
        "cib_gb_046",
        "cib_gb_048",
        "cib_gb_049",
        "cib_gb_050",
        "cib_gb_052",
        "cib_gb_053",
    ]
    test_id_gb = [
        "cib_gb_001",
        "cib_gb_003",
        "cib_gb_004",
        "cib_gb_010",
        "cib_gb_013",
        "cib_gb_018",
        "cib_gb_022",
        "cib_gb_024",
        "cib_gb_025",
        "cib_gb_026",
        "cib_gb_030",
        "cib_gb_037",
        "cib_gb_040",
        "cib_gb_047",
        "cib_gb_051",
        "cib_gb_054",
        "cib_gb_055",
    ]

    experiment = RuleInferenceExperiment(
        config_path="examples/cib_mp/agent/",
        original_config_file="cwd_agent_prompt_template.yaml",
        test_data_file="examples/cib_mp/examples.yaml",
        output_file_path="temp/rule_pruning_cibmp_20250623_2",
        max_iteration=3,
    )
    # experiment.load_test_data(train_id_gb)
    # asyncio.run(experiment.tune_question_batch())
    # experiment.experiment_result = results
    # combined_rules = asyncio.run(experiment.consolidate_rules())
    # print(combined_rules)
    experiment.load_test_data()
    asyncio.run(experiment.rule_pruning())
