from langchain.tools import tool
from typing import Annotated
from .utils import is_smiles, is_multiple_smiles, pubchem_query2smiles,query2cas,smiles2name
from .chemspace import ChemSpace
from .safety import ControlChemCheck

@tool
def Name2SMILES(
        query:Annotated[str,'Input a molecule name'],
        chemspace_api_key:Annotated[str,'your chemspace_api_key (may given in the system prompt)']=None,
    ):
    """
    Name2SMILES
        Input a molecule name, returns SMILES.
        Useful to get the SMILES string of one molecule by searching the name of a molecule. Only query with one specific name.
        This function queries the given molecule name and returns a SMILES string from the record

    Remark: Leave the chemspace_api_key=None if not existed
    """

    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    if is_smiles(query) and is_multiple_smiles(query):
        return "Multiple SMILES strings detected, input one molecule at a time."
    try:
        smi = pubchem_query2smiles(query, url)
    except Exception as e:
        if chemspace_api_key:
            try:
                chemspace = ChemSpace(chemspace_api_key)
                smi = chemspace.convert_mol_rep(query, "smiles")
                smi = smi.split(":")[1]
            except Exception:
                return str(e)
        else:
            return str(e)
        msg = "Note: " + ControlChemCheck(smi)
        if "high similarity" in msg or "appears" in msg:
            return f"CAS number {smi}found, but " + msg
        return smi

@tool
def Query2CAS(query:Annotated[str,'Input molecule (name or SMILES)']):
    """
    Mol2CAS
        Input molecule (name or SMILES), returns CAS number.
    """
    url_cid = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON"
    )
    url_data = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
    )
    def _run(self, query: str) -> str:
        try:
            # if query is smiles
            smiles = None
            if is_smiles(query):
                smiles = query
            try:
                cas = query2cas(query, self.url_cid, self.url_data)
            except ValueError as e:
                return str(e)
            if smiles is None:
                try:
                    smiles = pubchem_query2smiles(cas, None)
                except ValueError as e:
                    return str(e)
            # check if mol is controlled
            msg = ControlChemCheck(smiles)
            if "high similarity" in msg or "appears" in msg:
                return f"CAS number {cas}found, but " + msg
            return cas
        except ValueError:
            return "CAS number not found"

@tool
def SMILES2Name(
        query:Annotated[str,'Input SMILES'],
        chemspace_api_key:Annotated[str,'your chemspace_api_key (may given in the system prompt)']=None
):
    """
    SMILES2NAME
        Input SMILES, returns molecule name.
    """

    try:
        if not is_smiles(query):
            try:
                query = Name2SMILES(query, chemspace_api_key)
            except:
                raise ValueError("Invalid molecule input, no Pubchem entry")
        name = smiles2name(query, chemspace_api_key)
        # check if mol is controlled
        msg = "Note: " + ControlChemCheck(query)
        if "high similarity" in msg or "appears" in msg:
            return f"Molecule name {name} found, but " + msg
        return name
    except Exception as e:
        return "Error: " + str(e)
