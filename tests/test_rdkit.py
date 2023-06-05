import pytest
import rdkit.Chem as Chem

from chemcrow.tools.rdkit import *

@pytest.fixture
def singlemol():
    # Single mol
    return "O=C1N(C)C(C2=C(N=CN2C)N1C)=O"


@pytest.fixture
def molset1():
    # Set of mols
    return "O=C1N(C)C(C2=C(N=CN2C)N1C)=O.CC(C)c1ccccc1"

@pytest.fixture
def molset2():
    # Set of mols
    return "O=C1N(C)C(C2=C(N=CN2C)N1C)=O.O=C1N(C)C(C2=C(N=CN2C)N1CCC)=O"

@pytest.fixture
def single_iupac():
    # Test with a molecule with iupac name
    return "4-(4-hydroxyphenyl)butan-2-one"


@pytest.fixture
def common_name():
    # Test with a molecule with common name
    return "caffeine"


@pytest.fixture
def choline():
    # Test with a molecule in clintox
    return "CCCCCCCCC[NH+]1C[C@@H]([C@H]([C@@H]([C@H]1CO)O)O)O"


def test_molsim_1(molset1):
    tool = MolSimilarity()
    assert tool(molset1).endswith('not similar.')

def test_molsim_2(molset2):
    tool = MolSimilarity()
    assert tool(molset2).endswith('very similar.')

def test_molsim_same(singlemol):
    tool = MolSimilarity()
    out = tool("{}.{}".format(singlemol, singlemol))
    assert out == "Error: Input Molecules Are Identical"

def test_molsim_badinp(singlemol):
    tool = MolSimilarity()
    out = tool(singlemol)
    assert out == "Input error, please input two smiles strings separated by '.'"



#    largest_mol,
#    modify_mol,
#    patent_check,
#    smiles2name,
#    list_functional_groups,
#    is_smiles,
#    similarity_quantifier,
#)
#
#
#
#
#def test_largestmol(molset, cafeina):
#    largest = largest_mol(molset)
#    assert largest == cafeina
#
#
## Testing smiles2name function
#def test_smiles2name(cafeina):
#    name = smiles2name(cafeina)
#    assert name == "caffeine"
#
#
#def test_smiles2name_iupac(single_iupac):
#    name = smiles2name(single_iupac)
#    assert name == "Invalid SMILES string"
#
#
#def test_smilessimilarity(cafeina, choline):
#    message = similarity_quantifier(f"{cafeina}.{choline}")
#    assert (
#        message
#        == "The Tanimoto similarity between O=C1N(C)C(C2=C(N=CN2C)N1C)=O and CCCCCCCCC[NH+]1C[C@@H]([C@H]([C@@H]([C@H]1CO)O)O)O is 0.0189, indicating that the two molecules are not similar."
#    )
#
#
#def test_smilessimilarity_1smiles(cafeina):
#    # this should give an error
#    message = similarity_quantifier(f"{cafeina}.{cafeina}")
#    assert message == "Error: Input Molecules Are Identical"
#
#
#def test_smilessimilarity_iupac(cafeina, single_iupac):
#    # this should give an error
#    message = similarity_quantifier(f"{cafeina}.{single_iupac}")
#    assert message == "Error: Not a valid SMILES string"
#
#
#def test_patentcheck(cafeina):
#    patented = patent_check(cafeina)
#    assert patented == "Patented"
#
#
#def test_patentcheck_iupac(single_iupac):
#    # should give an error
#    patented = patent_check(single_iupac)
#    assert patented == "Invalid SMILES string"
#
#
#def test_patentcheck_not(choline):
#    patented = patent_check(choline)
#    assert patented == "Novel"
#
#
#
#def test_is_smiles(cafeina, single_iupac):
#    assert is_smiles(cafeina)
#    assert is_smiles(cafeina + "$@!") == False
#    assert is_smiles(single_iupac) == False
