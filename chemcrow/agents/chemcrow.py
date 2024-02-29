import os
from dotenv import load_dotenv
from typing import Optional, Dict
import langchain
import nest_asyncio
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from langchain.llms import GPT4All

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(model, temp, verbose, api_key, max_tokens=1000, n_ctx=2048):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        load_dotenv()
        try:
            llm = langchain.chat_models.ChatOpenAI(
                temperature=temp,
                model_name=model,
                request_timeout=1000,
                streaming=True if verbose else False,
                callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
                openai_api_key = api_key
            )
        except:
            raise ValueError("Invalid OpenAI API key")
    elif os.path.exists(model):
        ext = os.path.splitext(model)[-1].lower()
        if ext == ".gguf":
            # If GPT4All style weights
            llm = GPT4All(model=model, max_tokens=max_tokens, verbose=False)
        else:
            raise ValueError(f"Found file: {model}, however only models with .gguf format are suported currently.")
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm



class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-4-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: str = '',
        api_keys: Dict[str, str] = {},
        max_tokens: int = 1000, # Not required for using OpenAI's API
        n_ctx: int =  2048
    ):
        """Initialize ChemCrow agent."""

        self.llm = _make_llm(model, temp, verbose, openai_api_key, max_tokens, n_ctx)

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            tools_llm = _make_llm(tools_model, temp, verbose, openai_api_key, max_tokens, n_ctx)
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
        return outputs["output"]
