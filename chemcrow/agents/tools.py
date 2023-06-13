import os
import functools
from rdkit import Chem
import langchain
from langchain import agents, prompts, chains, llms
from langchain.tools.python.tool import PythonREPLTool

from chemcrow.tools import *

class ChemTools:
    def __init__(
            self,
            llm_T=0.1,
            llm='gpt-3.5-turbo',
            openai=None,
            serp=None,
    ):
        self.openai_key = os.getenv("OPENAI_API_KEY") or openai
        self.serp_key = os.getenv("SERP_API_KEY") or serp

        # Initialize standard tools
        llm = langchain.chat_models.ChatOpenAI(
            temperature=llm_T,
            model_name=llm
        )

        self.all_tools = agents.load_tools(
            ["python_repl", "human"], llm
        )

        self.all_tools += [
            Query2SMILES(),
            Query2CAS(),
            PatentCheck(),
            MolSimilarity(),
            SMILES2Weight(),
            FuncGroups(),
            LitSearch(),
            WebSearch(),
        ]


