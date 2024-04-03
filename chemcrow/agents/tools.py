import os
import inspect
from langchain import agents
from langchain.base_language import BaseLanguageModel

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
from langchain.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool

from chemcrow.tools import *


def make_tools(llm: BaseLanguageModel, api_keys: dict = {}, verbose=True):
    serp_api_key = api_keys.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )
    semantic_scholar_api_key = api_keys.get("SEMANTIC_SCHOLAR_API_KEY") or os.getenv(
        "SEMANTIC_SCHOLAR_API_KEY"
    )

    all_tools = agents.load_tools(
        [
            #"python_repl",
            # "ddg-search",
            "wikipedia",
            # "human"
        ]
    )

    all_tools += [
        Query2SMILES(chemspace_api_key),
        PythonREPLTool(),
        Query2CAS(),
        SMILES2Name(),
        PatentCheck(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        ExplosiveCheck(),
        ControlChemCheck(),
        SimilarControlChemCheck(),
        SafetySummary(llm=llm),
        # Scholar2ResultLLM(
        #     llm=llm,
        #     openai_api_key=openai_api_key,
        #     semantic_scholar_api_key=semantic_scholar_api_key
        # ),
    ]

    # Get absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the data.csv file relative to the script directory
    data_file_path = os.path.join(script_dir, '../tools/create/data.csv')

    # Add mds tools and hsp tool
    mds_classes = inspect.getmembers(mds, inspect.isclass)
    hsp_classes = inspect.getmembers(hsp, inspect.isclass)
    mds_tools = [cls for name, cls in mds_classes if issubclass(cls, BaseTool) and cls is not BaseTool]
    hsp_tools = [cls for name, cls in hsp_classes if issubclass(cls, BaseTool) and cls is not BaseTool]
    all_tools += [tool() for tool in mds_tools]
    all_tools += [tool() for tool in hsp_tools]

    if chemspace_api_key:
        all_tools += [GetMoleculePrice(chemspace_api_key)]
    if serp_api_key:
        all_tools += [WebSearch(serp_api_key)]
    if rxn4chem_api_key:
        all_tools += [
            RXNPredict(rxn4chem_api_key),
            RXNRetrosynthesis(rxn4chem_api_key, openai_api_key),
        ]

    return all_tools
