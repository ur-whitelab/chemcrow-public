import pytest
import rdkit.Chem as Chem

from chemcrow.tools.rdkit import (
    MolSimilarity,
    SMILES2Weight,
    FuncGroups
)

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



# MolSimilarity

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

def test_molsim_iupac(singlemol, single_iupac):
    tool = MolSimilarity()
    out = tool("{}.{}".format(singlemol, single_iupac))
    assert out == "Error: Not a valid SMILES string"

# SMILES2Weight

def test_mw(singlemol):
    tool = SMILES2Weight()
    mw = tool(singlemol)
    assert abs(mw - 194.) < 1.

def test_badinp(singlemol):
    tool = SMILES2Weight()
    mw = tool(singlemol + "x")
    assert mw == "Invalid SMILES string"


# FuncGroups

def test_fg_single(singlemol):
    tool = FuncGroups()
    out = tool(singlemol)
    assert 'ketones' in out

def test_fg_iupac(single_iupac):
    tool = FuncGroups()
    out = tool(single_iupac)
    assert out == "Wrong argument. Please input a valid molecular SMILES."

