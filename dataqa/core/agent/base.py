from langgraph.graph.graph import CompiledGraph

from dataqa.core.llm.base_llm import BaseLLM
from dataqa.core.memory import Memory


class Agent:
    name: str
    memory: Memory
    llm: BaseLLM
    workflow: CompiledGraph

    def __init__(self, memory: Memory, llm: BaseLLM):
        self.memory = memory
        self.llm = llm
        self.workflow = self.build_workflow(memory=memory, llm=llm)

    def build_workflow(self, memory: Memory, llm: BaseLLM) -> CompiledGraph:
        raise NotImplementedError

    def display_workflow(self, out_path):
        self.workflow.get_graph(xray=2).draw_mermaid_png(
            output_file_path=out_path
        )

    async def __call__(self, state):
        raise NotImplementedError