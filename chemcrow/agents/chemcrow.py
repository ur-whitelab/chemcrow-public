import langchain, os
import nest_asyncio
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor


from langchain.llms import LlamaCpp, GPT4All

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(model, temp, verbose, api_key, max_tokens=1000, n_ctx=2048):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
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
            return "Invalid openai key"
    elif model.startswith("text-"):
        try:
            llm = langchain.OpenAI(
                temperature=temp,
                model_name=model,
                streaming=True if verbose else False,
                callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
                openai_api_key = api_key
            )
        except:
            return "Invalid openai key"
    elif os.path.exists(model):
        ext = os.path.splitext(model)[-1].lower()
        if ext == ".bin":
            # Assuming this is a GPT4ALL style set of tensors
            llm = GPT4All(model=model, max_tokens=max_tokens, backend='gptj', verbose=False)
        elif ext == ".gguf":
            # Assuming this is a LlamaCpp style set of tensors
            llm = LlamaCpp(
                model_path=model,
                temperature=temp,
                max_tokens=max_tokens,
                n_ctx=n_ctx,
                top_p=1,
                verbose=True, # Verbose is required to pass to the callback manager
            )
        else:
            raise ValueError(f"Found file: {model}, but this function is only able to load .bin and .gguf models.")    
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
        openai_api_key: str = None,
        api_keys: dict = None,
        max_tokens: int = 1000, # Not required for using OpenAI's API
        n_ctx: int =  2048
    ):

        self.llm = _make_llm(model, temp, verbose, openai_api_key, max_tokens, n_ctx)
        
        if isinstance(self.llm, str):
            return self.llm
        
        if tools is None:
            tools_llm = _make_llm(tools_model, temp, max_tokens, verbose, openai_api_key)
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
