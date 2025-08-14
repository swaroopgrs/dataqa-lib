import argparse
import os

import pandas as pd
import yaml


def extract_test_results(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        if "test_result.yml" in files:
            test_result_path = os.path.join(root, "test_result.yml")
            with open(test_result_path, "r") as file:
                test_result = yaml.safe_load(file)
                use_case = test_result["use_case_config"]["name"]
                example_id = test_result["input_data"]["id"]
                question = test_result["input_data"]["question"]
                ground_truth_output = test_result["input_data"][
                    "ground_truth_output"
                ]

                for prediction in test_result["predictions"]:
                    run_id = prediction["run_id"]
                    llm_label = prediction["evaluation"]["llm_label"]
                    llm_judge_output = prediction["evaluation"][
                        "llm_judge_output"
                    ]
                    combined_response = prediction["combined_response"]
                    summary = prediction["summary"]
                    final_response = (
                        prediction["final_response"]["response"]
                        if prediction["final_response"]
                        else None
                    )

                    results.append(
                        {
                            "config_name": os.path.basename(
                                os.path.dirname(os.path.dirname(root))
                            ),
                            "Use Case": use_case,
                            "Example ID": example_id,
                            "Run ID": run_id,
                            "Question": question,
                            "Ground Truth Output": ground_truth_output,
                            "LLM Label": llm_label,
                            "LLM Judge Output": llm_judge_output,
                            "Combined Response": combined_response,
                            "Summary": summary,
                            "Final Response": final_response,
                        }
                    )
    return pd.DataFrame(results)


def extract_evaluation_results(base_dir):
    evaluations = []
    for root, dirs, files in os.walk(base_dir):
        if "evaluation.yml" in files:
            evaluation_path = os.path.join(root, "evaluation.yml")
            with open(evaluation_path, "r") as file:
                evaluation = yaml.safe_load(file)
                config_name = os.path.basename(root)

                # Extract fields for each use case
                for use_case in evaluation["accuracy"]:
                    if use_case not in ["macro", "micro"]:
                        evaluations.append(
                            {
                                "Config": config_name,
                                "Use Case": use_case,
                                "Accuracy": evaluation["accuracy"][use_case],
                                "Majority Frequency": evaluation[
                                    "majority_frequency"
                                ][use_case],
                                "P50 Latency": evaluation["p50_latency"][
                                    use_case
                                ],
                                "P90 Latency": evaluation["p90_latency"][
                                    use_case
                                ],
                                "P99 Latency": evaluation["p99_latency"][
                                    use_case
                                ],
                                "Reject Rate": evaluation["reject_rate"][
                                    use_case
                                ],
                            }
                        )
    return pd.DataFrame(evaluations)


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract and summarize test and evaluation results."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Path to the base directory containing output folders",
    )
    return parser.parse_args()


def main():
    args = get_args()
    base_dir = args.directory
    output_dir = os.path.abspath(base_dir)

    test_results_df = extract_test_results(base_dir)
    evaluation_results_df = extract_evaluation_results(base_dir)

    # Save the results to excel files or print them
    test_results_df.to_excel(
        os.path.join(output_dir, "test_results_summary.xlsx"), index=False
    )
    evaluation_results_df.to_excel(
        os.path.join(output_dir, "evaluation_summary.xlsx"), index=False
    )

    print("Test Results Summary:")
    print(test_results_df)
    print("\nEvaluation Summary:")
    print(evaluation_results_df)


if __name__ == "__main__":
    main()
