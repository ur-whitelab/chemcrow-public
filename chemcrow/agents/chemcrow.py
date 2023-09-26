import langchain, os
import nest_asyncio
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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


def make_local_llm(model_path, temp, n_ctx=1024):
    print("make_loacl_LLM:", os.path.abspath("."))
    # Callbacks support token-wise streaming
    #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_path,
        temperature=temp,
        max_tokens=100,
        n_ctx=n_ctx,
        top_p=1,
        verbose=True, # Verbose is required to pass to the callback manager
    )
    return llm


class ChemCrow:
    def __init__(
        self,
        model_path,
        tools=None,
        model="gpt-3.5-turbo-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        openai_api_key: str = None,
        api_keys: dict = None
    ):
# =============================================================================
#         try:
#             self.llm = _make_llm(model, temp, verbose, openai_api_key)
#         except:
#             return "Invalid openai key"
# 
# =============================================================================
        self.llm = make_local_llm(model_path, temp)
        
        if tools is None:
            tools_llm = make_local_llm(model_path, temp)
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
            #return_intermediate_steps=True,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    #nest_asyncio.apply()  # Fix "this event loop is already running" error

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs['output']
        # Parse long output (with intermediate steps)
        #intermed = outputs["intermediate_steps"]

        #final = ""
        #for step in intermed:
        #    final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
        #final += f"Final Answer: {outputs['output']}"

        #rephrased = self.rephrase_chain.run(question=prompt, agent_ans=final)
        #print(f"ChemCrow output: {rephrased}")
        #return rephrased
