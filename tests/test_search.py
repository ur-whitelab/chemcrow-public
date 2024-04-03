import ast
import os

import pytest
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from chemcrow.tools.search import PatentCheck, Scholar2ResultLLM
from chemcrow.utils import split_smiles

load_dotenv()


@pytest.fixture
def questions():
    qs = [
        "What are the effects of norhalichondrin B in mammals?",
    ]
    return qs[0]


@pytest.mark.skip(reason="This requires an api call")
def test_litsearch(questions):
    llm = ChatOpenAI()

    searchtool = Scholar2ResultLLM(llm=llm)
    for q in questions:
        ans = searchtool._run(q)
        assert isinstance(ans, str)
        assert len(ans) > 0
    if os.path.exists("../query"):
        os.rmdir("../query")


@pytest.fixture
def molset1():
    return "O=C1N(C)C(C2=C(N=CN2C)N1C)=O.CC(C)c1ccccc1"


@pytest.fixture
def singlemol():
    return "O=C1N(C)C(C2=C(N=CN2C)N1C)=O"


@pytest.fixture
def single_iupac():
    # Test with a molecule with iupac name
    return "4-(4-hydroxyphenyl)butan-2-one"


@pytest.fixture
def choline():
    # Test with a molecule in clintox
    return "CCCCCCCCC[NH+]1C[C@@H]([C@H]([C@@H]([C@H]1CO)O)O)O"


@pytest.fixture
def patentcheck():
    return PatentCheck()


def test_patentcheck(singlemol, patentcheck):
    patented = patentcheck._run(singlemol)
    patented = ast.literal_eval(patented)
    assert len(patented) == 1
    assert patented[singlemol] == "Patented"


def test_patentcheck_molset(molset1, patentcheck):
    patented = patentcheck._run(molset1)
    patented = ast.literal_eval(patented)
    mols = split_smiles(molset1)
    assert len(patented) == len(mols)
    assert patented[mols[0]] == "Patented"
    assert patented[mols[1]] == "Novel"


def test_patentcheck_iupac(single_iupac, patentcheck):
    patented = patentcheck._run(single_iupac)
    assert patented == "Invalid SMILES string"


def test_patentcheck_not(choline, patentcheck):
    patented = patentcheck._run(choline)
    patented = ast.literal_eval(patented)
    assert len(patented) == 1
    assert patented[choline] == "Novel"
