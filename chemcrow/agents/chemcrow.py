import langchain
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
from .prompts import FORMAT_INSTRUCTIONS, SUFFIX, QUESTION_PROMPT, REPHRASE_TEMPLATE
import nest_asyncio
from .tools import make_tools

from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
from langchain import PromptTemplate, OpenAI, chat_models, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def _make_llm(model, temp, verbose):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
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
    ):
        self.llm = _make_llm(model, temp, verbose)
        if tools is None:
            tools_llm = _make_llm(tools_model, temp, verbose)
            tools = make_tools(tools_llm, verbose=verbose)
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
            return_intermediate_steps=True,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    nest_asyncio.apply()  # Fix "this event loop is already running" error

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        # Parse long output (with intermediate steps)
        intermed = outputs["intermediate_steps"]

        final = ""
        for step in intermed:
            final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
        final += f"Final Answer: {outputs['output']}"

        rephrased = self.rephrase_chain.run(question=prompt, agent_ans=final)
        print(f"ChemCrow output: {rephrased}")
        return rephrased
