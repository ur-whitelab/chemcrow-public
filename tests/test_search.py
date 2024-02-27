import os

import pytest
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from chemcrow.tools.search import Scholar2ResultLLM

load_dotenv()


@pytest.fixture
def questions():
    qs = [
        "What are the effects of norhalichondrin B in mammals?",
    ]
    return qs[0]


@pytest.mark.skip(reason="This test requires an API key and takes a long time to run.")
def test_litsearch(questions):
    llm = ChatOpenAI()

    searchtool = Scholar2ResultLLM(llm=llm)
    for q in questions:
        ans = searchtool._run(q)
        assert isinstance(ans, str)
        assert len(ans) > 0
    if os.path.exists("../query"):
        os.rmdir("../query")
