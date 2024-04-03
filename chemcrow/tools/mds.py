
import requests
from langchain.tools import BaseTool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

def get_BP(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_BP' in data:
                    BP = data['ACD_BP']
                else:
                    return "ACD_BP is not available for the SMILES in the MDS Database"
                return "BP = " + str(BP)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2BP(BaseTool):
    name = "SMILES2BP"
    description = "Input SMILES and get the boiling point"

    def __init__(self):
        super(SMILES2BP, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_BP(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_Enthalpy(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Enthalpy' in data:
                    Enthalpy = data['ACD_Enthalpy']
                else:
                    return "ACD_Enthalpy is not available for the SMILES in the MDS Database"
                return "Enthalpy = " + str(Enthalpy)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2Enthalpy(BaseTool):
    name = "SMILES2Enthalpy"
    description = "Input SMILES and get the enthalpy"

    def __init__(self):
        super(SMILES2Enthalpy, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_Enthalpy(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_FlashPt(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_FP' in data:
                    FlashPt = data['ACD_FP']
                else:
                    return "ACD_FP is not available for the SMILES in the MDS Database"
                return "FlashPt = " + str(FlashPt)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2FlashPt(BaseTool):
    name = "SMILES2FlashPt"
    description = "Input SMILES and get the flashpoint"

    def __init__(self):
        super(SMILES2FlashPt, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_FlashPt(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_InSol(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Intrinsic_Sol' in data:
                    InSol = data['ACD_Intrinsic_Sol']
                else:
                    return "ACD_Intrinsic_Sol is not available for the SMILES in the MDS Database"
                return "InSol = " + str(InSol)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2InSol(BaseTool):
    name = "SMILES2InSol"
    description = "Input SMILES and get the intrinsic solubility in water"

    def __init__(self):
        super(SMILES2InSol, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_InSol(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_LogPCons(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_LogP_v_LogP_Consensus' in data:
                    LogPCons = data['ACD_LogP_v_LogP_Consensus']
                else:
                    return "ACD_LogP_v_LogP_Consensus is not available for the SMILES in the MDS Database"
                return "LogPCons = " + str(LogPCons)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2LogPCons(BaseTool):
    name = "SMILES2LogPCons"
    description = "Input SMILES and get the logP consensus"

    def __init__(self):
        super(SMILES2LogPCons, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_LogPCons(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_Density(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_Density' in data:
                    Density = data['ACD_Prop_Density']
                else:
                    return "ACD_Prop_Density is not available for the SMILES in the MDS Database"
                return "Density = " + str(Density)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2Density(BaseTool):
    name = "SMILES2Density"
    description = "Input SMILES and get the density"

    def __init__(self):
        super(SMILES2Density, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_Density(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_IdxRef(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_Index_Of_Refraction' in data:
                    IdxRef = data['ACD_Prop_Index_Of_Refraction']
                else:
                    return "ACD_Prop_Index_Of_Refraction is not available for the SMILES in the MDS Database"
                return "IdxRef = " + str(IdxRef)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2IdxRef(BaseTool):
    name = "SMILES2IdxRef"
    description = "Input SMILES and get the index of refraction"

    def __init__(self):
        super(SMILES2IdxRef, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_IdxRef(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_MolWt(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_MW' in data:
                    MolWt = data['ACD_Prop_MW']
                else:
                    return "ACD_Prop_MW is not available for the SMILES in the MDS Database"
                return "MolWt = " + str(MolWt)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2MolWt(BaseTool):
    name = "SMILES2MolWt"
    description = "Input SMILES and get the molecular weight"

    def __init__(self):
        super(SMILES2MolWt, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_MolWt(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_MolRef(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_Molar_Refractivity' in data:
                    MolRef = data['ACD_Prop_Molar_Refractivity']
                else:
                    return "ACD_Prop_Molar_Refractivity is not available for the SMILES in the MDS Database"
                return "MolRef = " + str(MolRef)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2MolRef(BaseTool):
    name = "SMILES2MolRef"
    description = "Input SMILES and get the molar refractivity"

    def __init__(self):
        super(SMILES2MolRef, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_MolRef(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_MolVol(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_Molar_Volume' in data:
                    MolVol = data['ACD_Prop_Molar_Volume']
                else:
                    return "ACD_Prop_Molar_Volume is not available for the SMILES in the MDS Database"
                return "MolVol = " + str(MolVol)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2MolVol(BaseTool):
    name = "SMILES2MolVol"
    description = "Input SMILES and get the molar volume"

    def __init__(self):
        super(SMILES2MolVol, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_MolVol(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_Parachor(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_Parachor' in data:
                    Parachor = data['ACD_Prop_Parachor']
                else:
                    return "ACD_Prop_Parachor is not available for the SMILES in the MDS Database"
                return "Parachor = " + str(Parachor)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2Parachor(BaseTool):
    name = "SMILES2Parachor"
    description = "Input SMILES and get the parachor"

    def __init__(self):
        super(SMILES2Parachor, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_Parachor(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_Polariz(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_Polarizability' in data:
                    Polariz = data['ACD_Prop_Polarizability']
                else:
                    return "ACD_Prop_Polarizability is not available for the SMILES in the MDS Database"
                return "Polariz = " + str(Polariz)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2Polariz(BaseTool):
    name = "SMILES2Polariz"
    description = "Input SMILES and get the polarizability"

    def __init__(self):
        super(SMILES2Polariz, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_Polariz(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_SurfTen(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_Prop_Surface_Tension' in data:
                    SurfTen = data['ACD_Prop_Surface_Tension']
                else:
                    return "ACD_Prop_Surface_Tension is not available for the SMILES in the MDS Database"
                return "SurfTen = " + str(SurfTen)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2SurfTen(BaseTool):
    name = "SMILES2SurfTen"
    description = "Input SMILES and get the surface tension"

    def __init__(self):
        super(SMILES2SurfTen, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_SurfTen(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_VapPres25(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_VP_vp25' in data:
                    VapPres25 = data['ACD_VP_vp25']
                else:
                    return "ACD_VP_vp25 is not available for the SMILES in the MDS Database"
                return "VapPres25 = " + str(VapPres25)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2VapPres25(BaseTool):
    name = "SMILES2VapPres25"
    description = "Input SMILES and get the vapor pressure at 25C"

    def __init__(self):
        super(SMILES2VapPres25, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_VapPres25(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_VapPres32(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_VP_vp32' in data:
                    VapPres32 = data['ACD_VP_vp32']
                else:
                    return "ACD_VP_vp32 is not available for the SMILES in the MDS Database"
                return "VapPres32 = " + str(VapPres32)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2VapPres32(BaseTool):
    name = "SMILES2VapPres32"
    description = "Input SMILES and get the vapor pressure at 32C"

    def __init__(self):
        super(SMILES2VapPres32, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_VapPres32(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_LogL16(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_ABSOLV_LogL16' in data:
                    LogL16 = data['ACD_ABSOLV_LogL16']
                else:
                    return "ACD_ABSOLV_LogL16 is not available for the SMILES in the MDS Database"
                return "LogL16 = " + str(LogL16)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2LogL16(BaseTool):
    name = "SMILES2LogL16"
    description = "Input SMILES and get the Abraham solubility LogL16 (the logarithm of the solute gas phase dimensionless Ostwald partition coefficient into hexadecane at 298K)"

    def __init__(self):
        super(SMILES2LogL16, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_LogL16(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_McGVol(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_ABSOLV_McGowan_Volume' in data:
                    McGVol = data['ACD_ABSOLV_McGowan_Volume']
                else:
                    return "ACD_ABSOLV_McGowan_Volume is not available for the SMILES in the MDS Database"
                return "McGVol = " + str(McGVol)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2McGVol(BaseTool):
    name = "SMILES2McGVol"
    description = "Input SMILES and get the Abraham solubility McGowan Volume"

    def __init__(self):
        super(SMILES2McGVol, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_McGVol(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_SkinIrr(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'ACD_skin_irr' in data:
                    SkinIrr = data['ACD_skin_irr']
                else:
                    return "ACD_skin_irr is not available for the SMILES in the MDS Database"
                return "SkinIrr = " + str(SkinIrr)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2SkinIrr(BaseTool):
    name = "SMILES2SkinIrr"
    description = "Input SMILES and get the skin irritation probability"

    def __init__(self):
        super(SMILES2SkinIrr, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_SkinIrr(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_NumRotors(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'babel_num_rotors' in data:
                    NumRotors = data['babel_num_rotors']
                else:
                    return "babel_num_rotors is not available for the SMILES in the MDS Database"
                return "NumRotors = " + str(NumRotors)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2NumRotors(BaseTool):
    name = "SMILES2NumRotors"
    description = "Input SMILES and get the number of rotors (rotatable bonds. Useful for comparing flexibility of 2 molecules)"

    def __init__(self):
        super(SMILES2NumRotors, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_NumRotors(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_NumRings(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'babel_num_rings' in data:
                    NumRings = data['babel_num_rings']
                else:
                    return "babel_num_rings is not available for the SMILES in the MDS Database"
                return "NumRings = " + str(NumRings)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2NumRings(BaseTool):
    name = "SMILES2NumRings"
    description = "Input SMILES and get the number of rings"

    def __init__(self):
        super(SMILES2NumRings, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_NumRings(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_TPSA(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'babel_PSA' in data:
                    TPSA = data['babel_PSA']
                else:
                    return "babel_PSA is not available for the SMILES in the MDS Database"
                return "TPSA = " + str(TPSA)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2TPSA(BaseTool):
    name = "SMILES2TPSA"
    description = "Input SMILES and get the Topological Polar Surface Area (TPSA) of a molecule. TPSA is defined as the surface sum over all polar atoms or molecules"

    def __init__(self):
        super(SMILES2TPSA, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_TPSA(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_SynthAcc(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'rdkit_synthetic_accessibility' in data:
                    SynthAcc = data['rdkit_synthetic_accessibility']
                else:
                    return "rdkit_synthetic_accessibility is not available for the SMILES in the MDS Database"
                return "SynthAcc = " + str(SynthAcc)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2SynthAcc(BaseTool):
    name = "SMILES2SynthAcc"
    description = "Input SMILES and get the synthetic accessibility score"

    def __init__(self):
        super(SMILES2SynthAcc, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_SynthAcc(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()

def get_HOMOLUMOGap(smiles):
    endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/search/smiles/{smiles}"
    response = requests.get(endpoint_url)
    print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if 'molecule_ids' in data and len(data['molecule_ids']) > 0:
            id = data['molecule_ids'][0]['id']
            endpoint_url = f"http://cadmol.na.pg.com/tools/mds_api/fetch/molecule/properties/" + str(id)
            response = requests.get(endpoint_url)
            print(f"endpoint: {endpoint_url}, status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'homo_lumo' in data:
                    HOMOLUMOGap = data['homo_lumo']
                else:
                    return "homo_lumo is not available for the SMILES in the MDS Database"
                return "HOMOLUMOGap = " + str(HOMOLUMOGap)
            else:
                return "failed to retrieve data. http status code: {response.status_code}"
        else:
            return "molecule not in mds database"
    else:
        return "failed to retrieve data. http status code: {response.status_code}"

class SMILES2HOMOLUMOGap(BaseTool):
    name = "SMILES2HOMOLUMOGap"
    description = "Input SMILES and get the HOMO/LUMO band gap"

    def __init__(self):
        super(SMILES2HOMOLUMOGap, self).__init__()

    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return get_HOMOLUMOGap(can_smiles)

    async def _arun(self, smiles: str) -> str:
        raise NotImplementedError()
