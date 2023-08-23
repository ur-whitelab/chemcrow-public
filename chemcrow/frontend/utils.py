import requests
from rdkit import Chem
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

def cdk(smiles):
    """
    Get a depiction of some smiles.
    """

    url = "https://www.simolecule.com/cdkdepict/depict/wob/svg"
    headers = {'Content-Type': 'application/json'}
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
        }
    )
    return response.text

def is_valid_smiles(smi):
    # First check if it's a molecule
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m: return True

    # Check if rxn
    reaction = Chem.rdChemReactions.ReactionFromSmarts(
        smi,
        useSmiles=True
    )
    if reaction: return True
    return False


tool_parse_pt = PromptTemplate(
    input_variables = ["input_tool"],
    template = """
You are a data parser. I will give you two texts, the input and output of a tool. Your task is to extract and reformat every chemical structure and reaction in the text.
Instructions:
- Only copy SMILES strings. If a molecule is not given as SMILES, don't use it.
- Do not fill in blanks with your knowledge.
- Return a JSON with the format {{"status":"<parsing-status>", "result":"<the-result-here>", "explaination":<argument-your-decision>}}
- "status" is "OK" if SMILES were found and correctly parsed, else return "NO SMILES".
- Do not separate SMILES into lists if they are separated by dot. Just copy it.

Example of SMILES:
- c1ccccc1
- [C@@H]1C=C2c3cccc4[nH]cc
- cc(c34)C[C@H]2N(C)C1

Input: {input_tool}
Begin!
"""
)

llm = ChatOpenAI(
    model='gpt-4',
    temperature=0.05
)
tool_parse_chain = LLMChain(
    prompt = tool_parse_pt,
    llm = llm
)
