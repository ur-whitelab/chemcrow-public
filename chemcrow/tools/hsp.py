import requests
from langchain.tools import BaseTool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

def get_hansen(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"Endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Check if 'molecule_ids' key exists and if it contains at least one element
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"Endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'hansen_solubility' in data:
                    HSP_D = data['hansen_solubility']['dispersion']
                    HSP_P = data['hansen_solubility']['polar']
                    HSP_H = data['hansen_solubility']['h_bonding']
                else:
                    return "hansen solubility is not available for the SMILES in mds database"
                return "HSP_D = " + str(HSP_D), "HSP_P = " + str(HSP_P),"HSP_H = " + str(HSP_H)
            else:
                return "Failed to retrieve data. HTTP status code: {response.status_code}"
        else:
            return "Molecule not in MDS database"
    else:
        return "Failed to retrieve data. HTTP status code: {response.status_code}"

class SMILES2HSP(BaseTool):
    name = "SMILES2HSP"
    description = "Input SMILES, returns Hansen solubility parameters (HSP_D: dispersion, HSP_P: polar, HSP_H: h_bonding). The distance between two molecules in HSP space is sqrt(4(HSP_D1-HSP_D2)² + (HSP_P1-HSP_P2)² + (HSP_H1-HSP_H2)²). This tool should not be used to predict absolute solubility. It can only be used to compare the relative solubilities of molecules in a solvent. Examples for where tool can be used: 1. Which is more or less soluble among molecule A,B,C in solvent D? 2. Which is more or less compatible among X,Y in Z? Examples where tool should not be used: 1. Is A soluble in B? 2. Is X compatible with Y?"
    #description = "Input SMILES, returns Hansen solubility parameters. The distance between two molecules in HSP space is 4(HSP_D1-HSP_D2)² + (HSP_P1-HSP_P2)² + (HSP_H1-HSP_H2)²."

    def __init__(
        self,
    ):
        super(SMILES2HSP, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_hansen(can_smiles)

    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
