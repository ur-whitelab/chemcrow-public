import pytest
from dotenv import load_dotenv
from chemcrow.tools.safety import SafetySummary, ExplosiveCheck, ControlChemCheck

load_dotenv()


smiles = {
    "CCOCC(=O)OC": 'low',
    "CP(=O)(F)OC1CCCCC1": 'low',
    'O=P(Cl)(Cl)Cl': 'high',
    'OCCN(CCO)CCO': 'high',
    'OC(Cl)CN(CCO)CCO': 'high',
}
molecs_smi = [(d[0], d[1]) for d in smiles.items()]


cas = {
    "10025-87-3": True,
    "756-79-6": True,
    '10545-99-0': True,
    '58-08-2': False,
    '134523-00-5': False,
    '55-63-0': False,
}
molecs_cas = [(d[0], d[1]) for d in cas.items()]

@pytest.mark.parametrize("inp, expect", molecs_smi)
def test_controlchemcheck_smi(inp, expect):
    """Test safety measures on some test molecules in smiles."""

    tool = ControlChemCheck()

    ans = tool(inp)
    assert isinstance(ans, str)
    assert f'has a {expect} similarity' in ans


@pytest.mark.parametrize("inp, expect", molecs_cas)
def test_controlchemcheck_cas(inp, expect):
    """Test safety measures on some test cas numbers."""

    tool = ControlChemCheck()

    ans = tool(inp)
    assert isinstance(ans, str)
    if expect:
        assert 'appears in a list' in ans
    else:
        assert f'has a low similarity' in ans

