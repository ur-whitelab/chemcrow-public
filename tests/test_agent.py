import os
import pytest
import chemcrow


def test_version():
    assert chemcrow.__version__

@pytest.mark.skip(reason="No API key")
def test_agent_init():
    chem_model = chemcrow.ChemCrow(
        model="gpt-4-0613",
        temp=0.1,
        max_iterations=2,
        api_keys={}
    )
    chem_model.run("What is the molecular weight of tylenol?")
