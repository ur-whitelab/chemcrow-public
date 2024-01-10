from rdkit import Chem
from rdkit import DataStructs


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
        fp1 = Chem.RDKFingerprint(Chem.MolFromSmiles(s1))
        fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles(s2))
        return DataStructs.FingerprintSimilarity(fp1, fp2)
    except Exception:
        return 0.0
