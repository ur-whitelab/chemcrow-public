import pytest
from dotenv import load_dotenv

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
