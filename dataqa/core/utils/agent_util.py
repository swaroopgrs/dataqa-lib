from enum import Enum
from typing import List, Literal

import pandas as pd

from dataqa.core.utils.dataframe_utils import df_to_markdown


class NodeName(Enum):
    planner = "planner"
    replanner = "replanner"
    retrieval_worker = "retrieval_worker"
    sql_generator = "sql_generator"
    sql_executor = "sql_executor"
    analytics_worker = "analytics_worker"
    plot_worker = "plot_worker"
    agent = "agent"
    tools = "tools"


def colored(
    text, color=None, attrs=None, mode: Literal["terminal", "text"] = "terminal"
):
    if mode == "terminal":
        # Define colored as termcolor is not available on jenkins
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
        }
        reset = "\033[0m"
        bold = "\033[1m"

        # Start with an empty string for attributes
        attr_code = ""

        # Add color if specified
        if color in colors:
            attr_code += colors[color]

        # Add bold attribute if specified
        if attrs and "bold" in attrs:
            attr_code += bold

        return f"{attr_code}{text}{reset}"
    else:
        return f"[{text}]"


def indented(text: str, indent: str = "    ") -> str:
    lines = text.split("\n")
    indented_lines = [indent + line for line in lines]
    indented_text = "\n".join(indented_lines)
    return indented_text


def format_plan(tasks: List) -> str:
    c = 1
    plan_list = []
    for task in tasks:
        worker_name = task.worker.value
        description = task.task_description
        plan_list.append(f"{c} - {worker_name}: {description}")
        c += 1
    return "\n".join(plan_list)


def format_tool_calls(tool_calls: List) -> str:
    formatted_tool_calls = []
    for tool_call in tool_calls:
        name = tool_call["name"]
        formatted_tool_call = f"{name}(\n"
        for k, v in tool_call["args"].items():
            formatted_tool_call += f'    {k}="{v}",\n'
        formatted_tool_call += ")"
        formatted_tool_calls.append(formatted_tool_call)
    return "\n".join(formatted_tool_calls)


def dataframe_to_llm_judge_string(df_name: str, df: pd.DataFrame):
    if df is None:
        return f"No dataframe found for {df_name} in memory."
    message = (
        f"  - dataframe_name: {df_name}\n"
        f"    size: {len(df)} rows, {len(df.columns)} columns\n"
        "    Rows:\n"
    )
    N_ROWS_TO_DISPLAY = 40
    if len(df) > N_ROWS_TO_DISPLAY:
        first_n_rows = df.head(N_ROWS_TO_DISPLAY // 2)
        last_n_rows = df.tail(N_ROWS_TO_DISPLAY // 2)

        ellipsis_row = pd.DataFrame({col: ["..."] for col in df.columns})
        df_to_display = pd.concat(
            [first_n_rows, ellipsis_row, last_n_rows], ignore_index=True
        )
    else:
        df_to_display = df
    display_rows = df_to_markdown(df_to_display)
    return message + "\n".join([f"    {s}" for s in display_rows.split("\n")])


def image_to_llm_judge_string(name: str, df: pd.DataFrame):
    return f"Image is created from below dataframe\n{dataframe_to_llm_judge_string(name, df)}"


def format_dataframes(dataframe_names: List[str], memory, config) -> str:
    formatted_dfs = []
    for df_output_name in dataframe_names:
        df_output = memory.get_dataframe(df_output_name, config)
        df_summary_string = dataframe_to_llm_judge_string(
            df_output_name, df_output
        )
        formatted_dfs.append(df_summary_string)
    return "\n".join(formatted_dfs)


def format_images(image_names: List[str], memory, config) -> str:
    formatted_dfs = []
    for image_name in image_names:
        df_plot_data = memory.get_image_data(image_name, config)
        df_summary_string = dataframe_to_llm_judge_string(
            image_name, df_plot_data
        )
        formatted_dfs.append(df_summary_string)
    return "\n".join(formatted_dfs)


class AgentResponseParser:
    """Used to extract debug information from events"""

    def __init__(self, events, memory, config):
        self.events = events
        self.memory = memory
        self.config = config
        self.replan_count = 0
        self.run_statistics = {}
        self.processed_events = []
        self.formatted_events = self.process_events("text")

    def fill_missing_prompt_for_steps(self, prompt: str):
        for pe in self.processed_events:
            if pe["llm_info"] is not None:
                if pe["llm_info"]["prompt"] is None:
                    pe["llm_info"]["prompt"] = prompt

    def process_event_step(self, event, count, output="terminal"):
        processed_step = {
            "step_type": None,  # llm, tool, summary
            "step_count": count,
            "llm_info": None,
            "node": None,
        }
        formatted_output = []
        node_name = list(event[1].keys())[0]
        parent_node = event[0]
        if parent_node:
            parent_node = parent_node[0].split(":")[0]
            formatted_output.append(
                colored(
                    f"step {count}: {parent_node} - {node_name}",
                    "green",
                    mode=output,
                )
            )
        else:
            formatted_output.append(
                colored(f"step {count}: {node_name}", "green", mode=output)
            )

        if node_name in [NodeName.planner.value, NodeName.replanner.value]:
            processed_step["step_type"] = "llm"
            processed_step["node"] = node_name
            processed_step["llm_info"] = {
                "input_token_count": event[1][node_name]["llm_output"][
                    0
                ].metadata.input_token,
                "output_token_count": event[1][node_name]["llm_output"][
                    0
                ].metadata.output_token,
                "time": event[1][node_name]["llm_output"][0].metadata.latency,
                "model": event[1][node_name]["llm_output"][0].metadata.model,
                "prompt": event[1][node_name]["llm_output"][0].prompt[0][
                    "content"
                ],
            }
            if "plan" in event[1][node_name] and event[1][node_name]["plan"]:
                formatted_output.append(
                    indented(format_plan(event[1][node_name]["plan"][0].tasks))
                )
                if node_name == NodeName.planner.value:
                    self.run_statistics["task_count_in_initial_plan"] = len(
                        event[1][node_name]["plan"][0].tasks
                    )
                else:
                    self.run_statistics["replan_count"] += 1
            else:
                formatted_output.append(
                    indented(
                        "Output message:"
                        + event[1][node_name]["final_response"].response
                    )
                )

                formatted_output.append(
                    indented(
                        "Output dataframe:"
                        + str(
                            event[1][node_name]["final_response"].output_df_name
                        )
                    )
                )
                df_summary_string = format_dataframes(
                    event[1][node_name]["final_response"].output_df_name,
                    self.memory,
                    self.config,
                )
                formatted_output.append(indented(df_summary_string))

                formatted_output.append(
                    indented(
                        "Output image:"
                        + str(
                            event[1][node_name][
                                "final_response"
                            ].output_img_name
                        )
                    )
                )
                df_image_string = format_images(
                    event[1][node_name]["final_response"].output_img_name,
                    self.memory,
                    self.config,
                )
                formatted_output.append(
                    indented(
                        "Image is created from below dataframe\n"
                        + df_image_string,
                        "      ",
                    )
                )
                self.run_statistics["replan_count"] += 1
        elif node_name == NodeName.sql_generator.value:
            processed_step["step_type"] = "llm"
            processed_step["node"] = node_name
            processed_step["llm_info"] = {
                "input_token_count": event[1][node_name]["llm_output"][
                    0
                ].metadata.input_token,
                "output_token_count": event[1][node_name]["llm_output"][
                    0
                ].metadata.output_token,
                "time": event[1][node_name]["llm_output"][0].metadata.latency,
                "model": event[1][node_name]["llm_output"][0].metadata.model,
                "prompt": event[1][node_name]["llm_output"][0].prompt[0][
                    "content"
                ],
            }
            formatted_output.append(
                indented(
                    "Reasoning:\n"
                    + event[1][node_name]["sql_generator_output"].reasoning
                )
            )
            formatted_output.append(
                indented(
                    "SQL:\n" + event[1][node_name]["sql_generator_output"].sql
                )
            )
        elif node_name == NodeName.sql_executor.value:
            processed_step["step_type"] = "tool"
            processed_step["node"] = node_name
            formatted_output.append(
                indented(event[1][node_name]["sql_executor_output"].dataframe)
            )
        elif node_name == NodeName.retrieval_worker.value:
            formatted_output.append(
                indented(
                    f"{node_name} summary:\n"
                    + event[1][node_name]["worker_response"]
                    .task_response[0]
                    .response
                )
            )
            df_output_name = event[1]["retrieval_worker"][
                "retrieval_worker_state"
            ][0].sql_executor_output.dataframe
            df_summary_string = format_dataframes(
                [df_output_name], self.memory, self.config
            )
            formatted_output.append(indented(df_summary_string))
            processed_step["step_type"] = "summary"
            processed_step["node"] = node_name
        elif node_name == NodeName.agent.value:
            if parent_node in [
                NodeName.plot_worker.value,
                NodeName.analytics_worker.value,
            ]:
                finish_reason = event[1]["agent"]["messages"][
                    0
                ].response_metadata["finish_reason"]
                if finish_reason == "tool_calls":
                    formatted_output.append(
                        indented(
                            "Tool call:\n"
                            + format_tool_calls(
                                event[1]["agent"]["messages"][0].tool_calls
                            )
                        )
                    )
                elif finish_reason == "stop":
                    formatted_output.append(
                        indented(
                            "Agent response:\n"
                            + str(event[1]["agent"]["messages"][0].content)
                        )
                    )
                else:
                    pass
                processed_step["step_type"] = "llm"
                processed_step["node"] = node_name
                processed_step["llm_info"] = {
                    "input_token_count": event[1][node_name]["messages"][
                        0
                    ].usage_metadata["input_tokens"],
                    "output_token_count": event[1][node_name]["messages"][
                        0
                    ].usage_metadata["output_tokens"],
                    "time": float(
                        event[1][node_name]["messages"][0].response_metadata[
                            "headers"
                        ]["cmp-upstream-response-duration"]
                    )
                    / 1000,
                    "model": event[1][node_name]["messages"][
                        0
                    ].response_metadata["headers"]["x-ms-deployment-name"],
                    "prompt": None,
                }
        elif node_name == NodeName.tools.value:
            if parent_node in [
                NodeName.plot_worker.value,
                NodeName.analytics_worker.value,
            ]:
                tool_name = event[1][node_name]["messages"][0].name
                tool_message = event[1][node_name]["messages"][0].content
                formatted_output.append(
                    indented(f"Tool ({tool_name}) message:\n{tool_message}")
                )
            self.run_statistics["tool_call_count"] += 1
            processed_step["step_type"] = "tool"
            processed_step["node"] = node_name
        elif node_name in [
            NodeName.plot_worker.value,
            NodeName.analytics_worker.value,
        ]:
            formatted_output.append(
                indented(
                    f"{node_name} summary:\n"
                    + event[1][node_name]["worker_response"]
                    .task_response[0]
                    .response
                )
            )
            prompt = event[1][node_name][f"{node_name}_state"][0].messages[0][
                "content"
            ]
            self.fill_missing_prompt_for_steps(prompt)
            processed_step["step_type"] = "summary"
            processed_step["node"] = node_name
        else:
            pass
        if output == "text":
            self.processed_events.append(processed_step)
        return "\n".join(formatted_output)

    def process_events(self, output="text"):
        self.run_statistics = {
            "task_count_in_initial_plan": None,
            "replan_count": 0,
            "tool_call_count": 0,
            "llm_stat": None,
        }
        formatted_events = []
        count = 1
        for event in self.events:
            formatted_events.append(
                self.process_event_step(event, count, output)
            )
            count += 1
        llm_stat = []
        for pe in self.processed_events:
            if pe["step_type"] == "llm":
                llm_stat.append(
                    [
                        f"step {pe['step_count']} - {pe['node']}",
                        pe["llm_info"]["input_token_count"],
                        pe["llm_info"]["output_token_count"],
                        pe["llm_info"]["time"],
                        pe["llm_info"]["model"],
                    ]
                )
        total_input_token = sum([x[1] for x in llm_stat])
        total_output_token = sum([x[2] for x in llm_stat])
        total_llm_time = sum([x[3] for x in llm_stat])
        llm_stat.append(
            ["total", total_input_token, total_output_token, total_llm_time, ""]
        )
        df_llm_stat = pd.DataFrame(
            llm_stat,
            columns=["step", "input_token", "output_token", "time", "model"],
        )
        self.run_statistics["llm_stat"] = df_llm_stat.to_markdown()
        return formatted_events

    def pretty_print_output(self):
        print("\n".join(self.process_events("terminal")))
        print(f"\nRun statistics:")
        for k, v in self.run_statistics.items():
            print(f"\t{k}: {v}") if k != "llm_stat" else print(
                indented(f"{k}:\n{v}")
            )

    def get_text_output(self, include_prompt=False):
        output = "\n".join(self.formatted_events)
        output += "\nRun statistics:\n"
        for k, v in self.run_statistics.items():
            output += (
                f"\t{k}: {v}" if k != "llm_stat" else indented(f"{k}:\n{v}")
            )
        if include_prompt:
            output += "\nPrompt for LLM steps:\n"
            for pe in self.processed_events:
                if pe["step_type"] == "llm":
                    output += f"step - {pe['step_count']}\n"
                    output += indented(str(pe["llm_info"]["prompt"])) + "\n\n"
        return output

    def get_prompt_for_step(self, step):
        return self.processed_events[step - 1]["llm_info"]["prompt"]

    def extract_steps_from_streaming_events(self) -> list[dict]:
        """extract the events that contains input/output of each node"""
        current_node = ""
        node_list = [
            "planner",
            "agent",
            "replanner",
            "tools",
            "retrieval_worker",
            "analytics_worker",
            "plot_worker",
        ]
        name_list = ["AzureChatOpenAI"]
        output = []
        i = 1
        c = 0
        for response in self.events:
            event = response["event"]
            name = response["name"]
            node = response.get("metadata", {}).get("langgraph_node", None)
            if node in node_list:
                if node != current_node:
                    output.append(
                        {
                            "sequence": i,
                            "node": node,
                            "raw": [],
                            "openai": [],
                        }
                    )
                    current_node = node
                    i += 1
                else:
                    pass

                if event in ["on_chain_end", "on_chat_model_end"]:
                    if name in node_list:
                        output[-1]["raw"].append(response)
                    elif name in name_list:
                        output[-1]["openai"].append(response)
            c += 1
        return output

    @staticmethod
    def process_step_of_streaming(step: dict) -> tuple[str, list]:
        """
        :param step: single step (node or tool)
        :return: tuple of output message as string, and list of input prompts
        """

        def get_tool_args_str(args_dict):
            tool_params_str = "\n".join(
                f"{key}: {value}" for key, value in args_dict.items()
            )
            return tool_params_str

        node = step.get("node", "")
        step_seq = step.get("sequence")
        output_msg = None
        prompt = None
        step_msg = f"Step {step_seq}; Node {node}:\n"
        output = step["raw"][0]["data"].get("output", None)
        if output is None:
            return "", []
        match node:
            case "orchestrator":
                output_resp = output.get("response", "")
                output_obj = output.get("objective", "")
                output_bsl = output.get("business_line", "")
                output_msg = f"Objective: {output_obj}\nBusiness line: {output_bsl}\nResponse: {output_resp}\n"
                prompt = step["openai"][0]["data"]["input"]["messages"]
            case "planner":
                output_plan = output.get("plan", "")
                output_msg = f"Plan: {output_plan}\n"
                prompt = step["openai"][0]["data"]["input"]["messages"]
            case "agent":
                output_msg = output.get("messages")[0].content
                output_msg = f"Agent output message: {output_msg}\n"
                prompt = step["openai"][0]["data"]["input"]["messages"]
            case "replan":
                output_plan = output.get("plan", "")
                output_rsp = output.get("response", "")
                output_msg = (
                    f"Updated plan: {output_plan}\nResponse: {output_rsp}\n"
                )
                prompt = step["openai"][0]["data"]["input"]["messages"]
            case "tools":
                tool_msg = output.get("messages")[0].content
                tool_calls = step["raw"][0]["data"]["input"]["messages"][
                    1
                ].tool_calls
                tool_call_msg = "\n".join(
                    [
                        f"{tc['name']}:\n{get_tool_args_str(tc['args'])}"
                        for tc in tool_calls
                    ]
                )
                output_msg = (
                    f"Tool message: {tool_msg}\nTool call: {tool_call_msg}\n"
                )
                prompt = None
        if prompt is not None:
            prompt_list = []
            for p in prompt[0]:
                prompt_list.append([type(p).__name__, p.content])
        else:
            prompt_list = None
        return step_msg + output_msg, prompt_list

    def output_steps_of_streaming(self) -> tuple[str, dict]:
        """
        combine output of all steps
        :return: tuple of combined output messages as string, and dictionary of prompts of all nodes
        """
        all_msg = ""
        all_prompt = {}
        for step in self.steps:
            output_msg, prompt = self.process_step(step)
            all_msg += output_msg
            all_prompt[step["sequence"]] = {
                "node": step["node"],
                "prompt": prompt,
            }
        return all_msg, all_prompt


# if __name__ == "__main__":
#     events_loaded = pickle.load(open("./temp/agent_events_2.pkl", "rb"))
#     agent_response = AgentResponseParser(events_loaded)
#     agent_response.process_events()
#     agent_response.pretty_print_output()

