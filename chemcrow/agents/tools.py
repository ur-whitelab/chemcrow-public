import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from chemcrow.tools import *


def make_tools(
        llm: BaseLanguageModel,
        api_keys: dict = {},
        verbose=True
):
    serp_key = api_keys.get('SERP_API_KEY') or os.getenv("SERP_API_KEY")
    rxn4chem_api_key = api_keys.get('RXN4CHEM_API_KEY') or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get('OPENAI_API_KEY') or os.getenv("OPENAI_API_KEY")

    all_tools = agents.load_tools([
        "python_repl",
        # "ddg-search",
        "wikipedia",
        # "human"
    ])

    all_tools += [
        Query2SMILES(),
        Query2CAS(),
        PatentCheck(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        ExplosiveCheck(),
        ControlChemCheck(),
        SimilarControlChemCheck(),
        SafetySummary(llm=llm),
        #LitSearch(llm=llm, verbose=verbose),
    ]
    if rxn4chem_api_key:
        all_tools += [
            RXNPredict(rxn4chem_api_key),
            RXNRetrosynthesis(rxn4chem_api_key, openai_api_key)
        ]

    return all_tools
