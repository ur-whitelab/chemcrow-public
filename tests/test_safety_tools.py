import pytest
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from chemcrow.tools.safety import ControlChemCheck, ExplosiveCheck, SafetySummary

load_dotenv()


smiles = {
    "CCOCC(=O)OC": "low",
    "CP(=O)(F)OC1CCCCC1": "low",
    "O=P(Cl)(Cl)Cl": "high",
    "OCCN(CCO)CCO": "high",
    "OC(Cl)CN(CCO)CCO": "high",
}
molecs_smi = [(d[0], d[1]) for d in smiles.items()]


cas = {
    "10025-87-3": True,
    "756-79-6": True,
    "10545-99-0": True,
    "58-08-2": False,
    "134523-00-5": False,
    "55-63-0": False,
}
molecs_cas = [(d[0], d[1]) for d in cas.items()]


@pytest.mark.parametrize("inp, expect", molecs_smi)
def test_controlchemcheck_smi(inp, expect):
    """Test safety measures on some test molecules in smiles."""

    tool = ControlChemCheck()

    ans = tool(inp)
    assert isinstance(ans, str)
    assert f"has a {expect} similarity" in ans


@pytest.mark.parametrize("inp, expect", molecs_cas)
def test_controlchemcheck_cas(inp, expect):
    """Test safety measures on some test cas numbers."""

    tool = ControlChemCheck()

    ans = tool(inp)
    assert isinstance(ans, str)
    if expect:
        assert "appears in a list" in ans
    else:
        assert f"has a low similarity" in ans


@pytest.mark.skip(reason="This requires an api call")
def test_safety_summary():
    llm = ChatOpenAI()
    safety_summary = SafetySummary(llm=llm)
    cas = "676-99-3"
    ans = safety_summary(cas)
    assert isinstance(ans, str)
    assert "valid CAS number" not in ans
    assert "not found" not in ans
    assert "operator safety" in ans.lower()
    assert "ghs" in ans.lower()
    assert "environment" in ans.lower()
    assert "societal" in ans.lower()


@pytest.fixture
def explosive():
    return ExplosiveCheck()


def test_explosive_check_exp(explosive):
    tnt_cas = "118-96-7"
    ans = explosive(tnt_cas)
    assert "Error" not in ans
    assert ans == "Molecule is explosive"


def test_explosive_check_nonexp(explosive):
    non_exp_cas = "10025-87-3"
    ans = explosive(non_exp_cas)
    assert "Error" not in ans
    assert ans == "Molecule is not known to be explosive"
