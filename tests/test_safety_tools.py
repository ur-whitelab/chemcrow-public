import pytest
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from chemcrow.tools.safety import ControlChemCheck, ExplosiveCheck, SafetySummary

load_dotenv()


@pytest.fixture
def controlledchemcheck():
    return ControlChemCheck()


def test_controlchemcheck_controlled(controlledchemcheck):
    ans_cas = controlledchemcheck._run("10025-87-3")
    ans_smi = controlledchemcheck._run("O=P(Cl)(Cl)Cl")
    assert "appears in a list" in ans_cas
    assert "appears in a list" in ans_smi


def test_controlchemcheck_notsimilar(controlledchemcheck):
    acetone_smi = "CC(=O)C"
    acetone_cas = "67-64-1"
    ans_cas = controlledchemcheck._run(acetone_cas)
    ans_smi = controlledchemcheck._run(acetone_smi)
    print("ans_cas", ans_cas)
    print("ans_smi", ans_smi)
    assert "appears in a list" not in ans_cas
    assert "appears in a list" not in ans_smi

    assert "is similar to" not in ans_cas
    assert "is similar to" not in ans_smi


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
