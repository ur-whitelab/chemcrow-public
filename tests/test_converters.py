import os

import pytest

from chemcrow.tools.chemspace import ChemSpace, GetMoleculePrice
from chemcrow.tools.converters import Query2CAS, Query2SMILES, SMILES2Name
from chemcrow.utils import canonical_smiles


@pytest.fixture
def singlemol():
    return "O=C1N(C)C(C2=C(N=CN2C)N1C)=O"


@pytest.fixture
def molset1():
    return "O=C1N(C)C(C2=C(N=CN2C)N1C)=O.CC(C)c1ccccc1"


@pytest.fixture
def single_iupac():
    # Test with a molecule with iupac name
    return "4-(4-hydroxyphenyl)butan-2-one"


def test_q2cas_iupac(single_iupac):
    tool = Query2CAS()
    out = tool._run(single_iupac)
    assert out == "5471-51-2"


def test_q2cas_cafeine(singlemol):
    tool = Query2CAS()
    out = tool._run(singlemol)
    assert out == "58-08-2"


def test_q2cas_badinp():
    tool = Query2CAS()
    out = tool._run("nomol")
    assert out.endswith("no Pubchem entry") or out.endswith("not found")


def test_q2s_iupac(single_iupac):
    tool = Query2SMILES()
    out = tool._run(single_iupac)
    assert out == "CC(=O)CCc1ccc(O)cc1"


def test_q2s_cafeine(singlemol):
    tool = Query2SMILES()
    out = tool._run("caffeine")
    assert out == canonical_smiles(singlemol)


def test_q2s_fail(molset1):
    tool = Query2SMILES()
    out = tool._run(molset1)
    assert out.endswith("input one molecule at a time.")


def test_getmolprice_no_api():
    tool = GetMoleculePrice(chemspace_api_key=None)
    price = tool._run("caffeine")
    assert "No Chemspace API key found" in price

def test_getmolprice(singlemol):
    if os.getenv("CHEMSPACE_API_KEY") is None:
        pytest.skip("No Chemspace API key found")
    else:
        tool = GetMoleculePrice(chemspace_api_key=os.getenv("CHEMSPACE_API_KEY"))
        price = tool._run(singlemol)
        assert "of this molecule cost" in price


def test_query2smiles_chemspace(singlemol, single_iupac):
    if os.getenv("CHEMSPACE_API_KEY") is None:
        pytest.skip("No Chemspace API key found")
    else:
        chemspace = ChemSpace(chemspace_api_key=os.getenv("CHEMSPACE_API_KEY"))
        smiles_from_chemspace = chemspace.convert_mol_rep("caffeine", "smiles")
        assert "CN1C=NC2=C1C(=O)N(C)C(=O)N2[13CH3]" in smiles_from_chemspace

        price = chemspace.buy_mol(singlemol)
        assert "of this molecule cost" in price

        price = chemspace.buy_mol(single_iupac)
        assert "of this molecule cost" in price


def test_smiles2name():
    smiles2name = SMILES2Name()
    assert (
        smiles2name.run("CN1C=NC2=C1C(=O)N(C)C(=O)N2[13CH3]")
        == "1,7-Dimethyl-3-(113C)methylpurine-2,6-dione"
    )
    assert "acetic acid" in smiles2name.run("CC(=O)O").lower()
    assert "Error:" in smiles2name.run("nonsense")
