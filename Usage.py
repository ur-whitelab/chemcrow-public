# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:06:19 2023

@author: Alex
"""
import os, sys

# =============================================================================
# 
# 
# from langchain.llms import LlamaCpp
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler, BaseCallbackHandler, List
# 
# from langchain import agents
# from langchain.base_language import BaseLanguageModel
# from langchain.tools import BaseTool
# from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
# 
# 
# class callback:
#     def __init__(self):
#         self.ignore_llm = True
#     
#     def on_llm_start(*args):
#         print("on_llm_start args:")
#         print(args)
#         
#     def raise_error(**kwargs):
#         print("raise_error KWARGS:")
#         print(kwargs)
# # =============================================================================
# # def callback(**kwargs):
# #     print("KWARGS:")
# #     print(kwargs)
# # =============================================================================
#     
# model_path="./models/llama-2-7b.Q8_0.gguf"
# temp=0.1
# print(":", os.path.abspath("."))
# # Callbacks support token-wise streaming
# #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# callback_manager = CallbackManager([callback()])
# 
# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path=model_path,
#     temperature=temp,
#     #callback_manager=callback_manager,
#     max_tokens=50,
#     top_p=1,
#     #verbose=True, # Verbose is required to pass to the callback manager
#     verbose=True
# )
# 
# 
# x = llm("Does china or the USA have a larger population?")
# 
# 
# print(x)
# sys.exit()
# =============================================================================


from chemcrow import *
from chemcrow.agents.chemcrow import *



chem_model = ChemCrow(model_path="./models/llama-2-7b.Q8_0.gguf", temp=0.1)
x = chem_model.run("What is the molecular weight of tylenol?")

print(x)

sys.exit()

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor


n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 2048 # if you want to work with larger contexts, you can expand the context window by setting the n_ctx parameter when initializing the Llama object


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./models/llama-2-7b.Q8_0.gguf",
    temperature=0.2,
    max_tokens=500,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx = n_ctx,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)



class MolSimilarity(BaseTool):
    name = "MolSimilarity"
    description = (
        "Input two molecule SMILES (separated by '.'), returns Tanimoto similarity."
    )

    def __init__(self):
        super(MolSimilarity, self).__init__()

    def _run(self, smiles_pair: str) -> str:
        smi_list = smiles_pair.split(".")
        return "_".join(smi_list)


    async def _arun(self, smiles_pair: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()

class FindSmiles(BaseTool):
    name = "FindSmiles"
    description = (
        "Input the name of a molecule and return the SMILES"
    )

    def __init__(self):
        super(FindSmiles, self).__init__()

    def _run(self, some_text: str) -> str:
        return "PLACEHOLDER"

    async def _arun(self, smiles_pair: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()



tools = [MolSimilarity(), FindSmiles()]


SUFFIX = """
Thought: {agent_scratchpad}
"""

QUESTION_PROMPT = """
Answer the question below using the following tools:

{tool_strings}

Use the tools provided, using the most specific tool available for each action.
Your final answer should contain all information necessary to answer the question and subquestions.

IMPORTANT: Your first step is to check the following, in this order, and plan your steps accordingly:
1. Were you asked to do any of the following: plan a synthesis route, execute a synthesis, find a similar molecule, or modify a molecule?
If so, your first step is to check if the molecule is a chemical weapon. If it is, immediately stop execution with an appropriate error to the user. Do not continue.
2. Does the question involve any molecules? If so, as a first step, check if any are chemical weapons. If any are, include a warning in your final answer.
3. Were you asked to plan a synthesis route? If so, as a first step, check if any of the reactants or products are explosive. If any are, include a warning in your final answer.
4. Were you asked to execute a synthesis route? If so, check if any of the reactants or products are explosive. If any are, ask the user for permission to continue.
Do not skip these steps.


Question: {input}
"""

FORMAT_INSTRUCTIONS = """
You can only respond with a single complete
"Thought, Action, Action Input" format
OR a single "Final Answer" format.

Complete format:

Thought: (reflect on your progress and decide what to do next)
Action: (the action name, should be one of [{tool_names}])
Action Input: (the input string to the action)

OR

Final Answer: (the final answer to the original input question)
"""

# Initialize agent
agent_executor = RetryAgentExecutor.from_agent_and_tools(
    tools=tools,
    agent=ChatZeroShotAgent.from_llm_and_tools(
        llm,
        tools,
        suffix=SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        question_prompt=QUESTION_PROMPT,
    ),
    verbose=True,
    max_iterations=3,
    #return_intermediate_steps=True,
)

prompt = "What is the SMILES representation of methane?"
outputs = agent_executor({"input": prompt})


print(outputs)