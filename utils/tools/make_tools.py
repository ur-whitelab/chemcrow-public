import os

from .chemspace import GetMoleculePrice
from .converters import Name2SMILES, Query2CAS, SMILES2Name
from .rk import MolSimilarity, SMILES2Weight, FuncGroups

from .safety import ExplosiveCheck, ControlChemCheck, SimilarControlChemCheck, SafetySummary
from .search import PatentCheck, WebSearch,LiteratureSearch

def make_tools(llm, api_keys: dict = {}, verbose=True):
    serp_api_key = api_keys.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )
    semantic_scholar_api_key = api_keys.get("SEMANTIC_SCHOLAR_API_KEY") or os.getenv(
        "SEMANTIC_SCHOLAR_API_KEY"
    )

    # all_tools = load_tools(
    #     [
    #         "python_repl",
    #         # "ddg-search",
    #         "wikipedia",
    #         # "human"
    #     ]
    # )
    all_tools=list()

    all_tools += [
        Name2SMILES,
        Query2CAS,
        SMILES2Name,
        PatentCheck,
        MolSimilarity,
        SMILES2Weight,
        FuncGroups,
        ExplosiveCheck,
        ControlChemCheck,
        SimilarControlChemCheck,
        # SafetySummary(),#to activate this,change the llm used in safety.py line 272
        LiteratureSearch,
    ]
    if chemspace_api_key:
        all_tools += [GetMoleculePrice(chemspace_api_key)]
    if serp_api_key:
        all_tools += [WebSearch(serp_api_key)]
    # if rxn4chem_api_key:
    #     all_tools += [
    #         RXNPredict(rxn4chem_api_key),
    #         RXNRetrosynthesis(rxn4chem_api_key, openai_api_key),
    #     ]

    return all_tools

if __name__ == '__main__':
    from langchain_openai import ChatOpenAI

    '''
    set up your llm here
    '''

    _llm = ChatOpenAI(
        api_key='ollama',
        model='qwen2.5:32b',
        base_url='http://192.168.31.194:8000/v1',
        temperature=0.2,
    )

    tools = make_tools(llm=_llm)
