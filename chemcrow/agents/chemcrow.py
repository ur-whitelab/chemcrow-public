import os
from dotenv import load_dotenv
from typing import Optional, Dict, Literal
import langchain
import nest_asyncio
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor


from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(
        model_type: Literal["openai", "tgi", "gpt4all"],
        model_server_url: Optional[str],
        verbose,
        api_key,
        **kwargs
):
    if model_type == "openai":
        load_dotenv()
        try:
            llm = langchain.chat_models.ChatOpenAI(
                temperature=kwargs['temp'],
                model_name=kwargs['model'],
                request_timeout=1000,
                streaming=True if verbose else False,
                callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
                openai_api_key = api_key
            )
        except:
            raise ValueError("Invalid OpenAI API key")

    elif model_type == "tgi":
        from langchain.llms import HuggingFaceTextGenInference
        llm = HuggingFaceTextGenInference(
            inference_server_url=model_server_url,
            max_new_tokens=kwargs['max_tokens'],
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=kwargs['temp'],
            repetition_penalty=1.03,
        )

    elif model_type == "gpt4all":
        from langchain.llms import GPT4All
        model = kwargs['model']
        if isinstance(model, str):
            if os.path.exists(model) and model.endswith(".gguf"):
                llm = GPT4All(
                    model=model,
                    max_tokens=kwargs['max_tokens'],
                    temp=kwargs['temp'],
                    verbose=False
                )
            else:
                raise ValueError(f"Couldn't load model. Only models with .gguf format are suported currently.")
        else:
            raise ValueError(f"Invalid model name: {model}")
    return llm



class ChemCrow:
    def __init__(
        self,
        model_type = 'openai',
        model_server_url: Optional[str] = None,
        tools=None,
        model="gpt-4-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_tokens: int = 4096,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: str = '',
        api_keys: Dict[str, str] = {},
    ):
        """Initialize ChemCrow agent."""

        self.llm = _make_llm(
            model_type,
            model_server_url,
            verbose,
            openai_api_key,
            model=model,
            max_tokens=max_tokens,
            temp=temp
        )

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            tools_llm = _make_llm(
                model_type,
                model_server_url,
                verbose,
                openai_api_key,
                model=model,
                max_tokens=max_tokens,
                temp=temp
            )
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
