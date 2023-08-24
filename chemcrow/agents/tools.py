import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from chemcrow.tools import *


def make_tools(llm: BaseLanguageModel, verbose=True):
    serp_key = os.getenv("SERP_API_KEY")

    all_tools = agents.load_tools(["python_repl"]) #, "human"])

    all_tools += [
        Query2SMILES(),
        Query2CAS(),
        PatentCheck(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        LitSearch(llm=llm, verbose=verbose),
    ]
    if serp_key:
        all_tools.append(WebSearch())
    return all_tools
