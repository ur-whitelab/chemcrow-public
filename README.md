[![tests](https://github.com/ur-whitelab/chemcrow/actions/workflows/tests.yml/badge.svg)](https://github.com/ur-whitelab/chemcrow) [![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg)](https://doi.org/10.48550/arXiv.2304.05376)




<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/chemcrow_dark_bold.png" width='100%'>
  <source media="(prefers-color-scheme: light)" srcset="./assets/chemcrow_light_bold.png" width='100%'>
  <img alt="ChemCrow logo" src="/assets/" width="100%">
</picture>


<br></br>


ChemCrow is an open source package for the accurate solution of reasoning-intensive chemical tasks.

It's built using Langchain and uses a collection of chemical tools from many sources, including IBM's RXN4Chemistry, RDKit, and paper-qa, as well as access to some relevant databases in chemistry, like Pubchem, chem-space, and others.


## 👩‍💻 Installation

```
conda create -n chemcrow python=3.8
conda activate chemcrow
pip install -e .
```

## 🔥 Usage
First set up your API keys in your environment. For the moment you need an OpenAI API key.

Other included tools also need API keys, particularly SERP (for web searches), RXN4Chem and chem-space. 

More examples can be found in the [notebooks](./notebooks/) folder.


```python
from chemcrow.agents import ChemTools, ChemCrow

chemtools = ChemTools()
chem_model = ChemCrow(
  # Get the toolset(s) that may be relevant:
    chemtools.mol_tools +
    chemtools.search_tools,

  # Specify the LLM
    model="gpt-4",
    temp=0.1,
)


# Now define your task
task = (
    "Tell me what the boiling point is of the reaction product between isoamyl acetate and ethanol."
    "To do this, predict the product of this reaction, and then find its boiling point. "
)

chem_model.run(task)
```

