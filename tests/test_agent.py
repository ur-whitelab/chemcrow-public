import os
from pydantic import ValidationError
import chemcrow


def test_version():
    assert chemcrow.__version__


def test_agent_init():

    try:
        chem_model = chemcrow.ChemCrow(
            model="gpt-4-0613",
            temp=0.1,
            max_iterations=2,
            api_keys={}
        )
        chem_model.run("What is the molecular weight of tylenol?")

    except ValueError:
        # Check if key in environment is valid
        oai_key = os.getenv('OPENAI_API_KEY')
        if oai_key:
            try:
                from langchain import OpenAI
                OpenAI(oai_key)
                assert False  # of this works, the error is in chemcrow
            except ValidationError:
                pass  # oai_key given but not valid

