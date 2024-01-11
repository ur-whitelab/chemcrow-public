import pytest
from dotenv import load_dotenv
from chemcrow.tools.safety import SafetySummary, ExplosiveCheck
from chemcrow.tools.databases import ControlChemCheck

load_dotenv()


data = {
    "CCOCC(=O)OC": 'low',
    "CP(=O)(F)OC1CCCCC1": 'low',
    'O=P(Cl)(Cl)Cl': 'high',
    'OCCN(CCO)CCO': 'high',
    'OC(Cl)CN(CCO)CCO': 'high',
}
molecs = [(d[0], d[1]) for d in data.items()]


@pytest.mark.parametrize("inp, expect", molecs)
def test_controlchemcheck(inp, expect):
    """Test safety measures on some test molecules."""

    tool = ControlChemCheck()

    ans = tool(inp)
    assert isinstance(ans, str)
    assert f'has a {expect} similarity' in ans

