import chemcrow


def test_version():
    assert chemcrow.__version__


def test_agent_init():
    chem_model = chemcrow.ChemCrow(model="gpt-4-0613", temp=0.1, max_iterations=2)
    chem_model.run("What are the effects of norhalichondrin B in mammals?")
