import pytest

from chemcrow.tools.databases import PatentCheck, Query2CAS, Query2SMILES
from chemcrow.utils import canonical_smiles


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


# Query2SMILES


def test_q2s_iupac(single_iupac):
    tool = Query2SMILES()
    out = tool(single_iupac)
    assert out == "CC(=O)CCc1ccc(O)cc1"


def test_q2s_cafeine(singlemol):
    tool = Query2SMILES()
    out = tool("caffeine")
    assert out == canonical_smiles(singlemol)


def test_q2s_fail(molset1):
    tool = Query2SMILES()
    out = tool(molset1)
    assert out.endswith("input one molecule at a time.")


# Query2CAS


def test_q2cas_iupac(single_iupac):
    tool = Query2CAS()
    out = tool(single_iupac)
    assert out == "5471-51-2"


def test_q2cas_cafeine(singlemol):
    tool = Query2CAS()
    out = tool(singlemol)
    assert out == "58-08-2"


def test_q2cas_badinp():
    tool = Query2CAS()
    out = tool("nomol")
    assert out.endswith("no Pubchem entry")


# PatentCheck


def test_patentcheck(singlemol):
    tool = PatentCheck()
    patented = tool(singlemol)
    assert patented == "Patented"


def test_patentcheck_iupac(single_iupac):
    tool = PatentCheck()
    patented = tool(single_iupac)
    assert patented == "Invalid SMILES string"


def test_patentcheck_not(choline):
    tool = PatentCheck()
    patented = tool(choline)
    assert patented == "Novel"
