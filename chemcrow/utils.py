from rdkit import Chem


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
    except Exception as e:
        return "Invalid SMILES string"
