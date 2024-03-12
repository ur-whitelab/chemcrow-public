import os

import pytest

import chemcrow


def test_version():
    assert chemcrow.__version__


@pytest.mark.skip(reason="This requires an api call")
def test_agent_init():
    chem_model = chemcrow.ChemCrow(
        model="gpt-3.5-turbo-0125", temp=0.1, max_iterations=2, api_keys={}
    )
    out = chem_model.run("hello")
    assert isinstance(out, str)
