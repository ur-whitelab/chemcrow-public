import requests
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from rdkit import Chem


def cdk(smiles):
    """
    Get a depiction of some smiles.
    """

    url = "https://www.simolecule.com/cdkdepict/depict/wob/svg"
    headers = {"Content-Type": "application/json"}
    response = requests.get(
        url,
        headers=headers,
        params={
            "smi": smiles,
            "annotate": "colmap",
            "zoom": 2,
            "w": 150,
            "h": 80,
            "abbr": "off",
        },
    )
    return response.text
