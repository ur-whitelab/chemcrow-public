import langchain
from pydantic import ValidationError
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
from typing import Optional

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(model, temp, verbose, api_key):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
            openai_api_key = api_key
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
            openai_api_key = api_key
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm

class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-3.5-turbo-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {}
    ):
        try:
            self.llm = _make_llm(model, temp, verbose, openai_api_key)
        except ValidationError:
            raise ValueError('Invalid OpenAI API key')

        if tools is None:
            tools_llm = _make_llm(tools_model, temp, verbose, openai_api_key)
            tools = make_tools(
                tools_llm,
                api_keys = api_keys,
                verbose=verbose
            )

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

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs['output']
