import pytest
import rdkit.Chem as Chem
from dotenv import load_dotenv
from chemcrow.tools.search import (
    LitSearch,
    WebSearch
)

load_dotenv()


@pytest.fixture
def questions():
    qs = [
        'What are the effects of norhalichondrin B in mammals?',
    ]
    return qs

def test_litsearch(questions):
    searchtool = LitSearch()

    for q in questions:
        ans = searchtool(q)
        assert isinstance(ans, str)
        assert len(ans)>0

def test_websearch(questions):
    searchtool = WebSearch('google')

    for q in questions:
        ans = searchtool(q)
        assert isinstance(ans, str)
        assert len(ans)>0
