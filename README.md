[![tests](https://github.com/ur-whitelab/chemcrow-public/actions/workflows/tests.yml/badge.svg)](https://github.com/ur-whitelab/chemcrow-public) [![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg)](https://doi.org/10.48550/arXiv.2304.05376)




<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/chemcrow_dark_bold.png" width='100%'>
  <source media="(prefers-color-scheme: light)" srcset="assets/chemcrow_light_bold.png" width='100%'>
  <img alt="ChemCrow logo" src="/assets/" width="100%">
</picture>


<br></br>


ChemCrow is an open source package for the accurate solution of reasoning-intensive chemical tasks.

Built with Langchain, it uses a collection of chemical tools including RDKit, paper-qa, as well as some relevant databases in chemistry, like Pubchem and chem-space.


## 👩‍💻 Installation

```
pip install chemcrow
```

## 🔥 Usage
First set up your API keys in your environment.
```
export OPENAI_API_KEY=your-openai-api-key
```

You can optionally use Serp API:

```
export SERP_API_KEY=your-serpapi-api-key
```

In a Python session:
```python
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(model="gpt-4-0613", temp=0.1, verbose=True)
chem_model.run("What is the molecular weight of tylenol?")
```

## ⚠️ Note

This package does not contain all the tools described in the [ChemCrow paper](https://arxiv.org/abs/2304.05376) because 
of API usage restrictions and dual-use concern. Over time we will move more tools to this repo (and we will accept PRs). 

## ✅ Citation
Bran, Andres M., et al. "ChemCrow: Augmenting large-language models with chemistry tools." arXiv preprint arXiv:2304.05376 (2023).

```bibtex
@article{bran2023chemcrow,
      title={ChemCrow: Augmenting large-language models with chemistry tools}, 
      author={Andres M Bran and Sam Cox and Oliver Schilter and Carlo Baldassari and Andrew D White and Philippe Schwaller},
      year={2023},
      eprint={2304.05376},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      publisher={arXiv}
}
```
