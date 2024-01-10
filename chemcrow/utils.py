from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False


def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]


def canonical_smiles(smiles):
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        return smi
    except Exception:
        return "Invalid SMILES string"


def tanimoto(s1, s2):
    """Calculate the Tanimoto similarity of two SMILES strings."""
    try:
        mol1 = Chem.MolFromSmiles(s1)
        mol2 = Chem.MolFromSmiles(s2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except (TypeError, ValueError, AttributeError):
        return "Error: Not a valid SMILES string"
