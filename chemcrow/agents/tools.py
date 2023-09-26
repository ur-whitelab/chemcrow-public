import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from chemcrow.tools import *


def make_tools(
        llm: BaseLanguageModel,
        api_keys: dict = {},
        verbose=True
):


    all_tools = [
        Query2SMILES(),
        Query2CAS(),
        PatentCheck(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        ExplosiveCheck(),
        SafetySummary(llm=llm),
        #LitSearch(llm=llm, verbose=verbose),
    ]


    return all_tools
