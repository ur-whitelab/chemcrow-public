from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from pydantic import Field


class ExceptionTool(BaseTool):
    name: str = Field(default="_Exception")
    description: str = Field(default="Exception tool")

    def _run(self, query: str) -> str:
        return query

    async def _arun(self, query: str) -> str:
        return query


class RetryAgentExecutor(AgentExecutor):
    """Agent executor that retries on output parser exceptions."""
    # for backwards compatibility
    handle_parsing_errors: bool = True
