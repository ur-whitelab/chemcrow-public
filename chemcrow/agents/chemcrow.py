import os

# import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from pydantic import ValidationError

from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from .dify_llm import DifyCustomLLM
from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_dify_llm(model, temp, api_key, base_url, streaming: bool = False):
    if  model.startswith("dify"):
        llm =DifyCustomLLM(api_key=api_key, user_id="llm-study-2", base_url=base_url)
    
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class ChemCrow:
    def __init__(
        self,
        tools=None,
        # model="gpt-4-0613",
        model="dify",
        tools_model="dify",
        # tools_model="gpt-4o-mini",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        api_keys: dict = {},
    ):
        """Initialize ChemCrow agent."""

        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path)
        dify_api_key = api_keys.get("DIFY_API_KEY") or os.getenv("DIFY_API_KEY")
        dify_base_url = api_keys.get("DIFY_BASE_URL") or os.getenv("DIFY_BASE_URL")
        try:
            self.llm = _make_dify_llm(model, temp, dify_api_key,dify_base_url, streaming)
        except ValidationError:
            raise ValueError("Invalid Dify API key")

        if tools is None:
            tools_llm = _make_dify_llm(tools_model, temp, dify_api_key,dify_base_url, streaming)
            tools = make_tools(tools_llm, api_keys=api_keys, verbose=verbose)

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt, callbacks=None):
        outputs = self.agent_executor({"input": prompt}, callbacks=callbacks)
        return outputs["output"]