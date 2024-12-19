
# Tools of organic chemistry

A docker container was prepared for each tool, which exposes an api for requests.

> docker run -d -p 8082:5000 doncamilom/rxnpred:latest

Where 5000 is fixed, and 8082 is the port to be exposed.

A request in curl can look like this

> curl -X POST -H "Content-Type: application/json" -d '{"smiles": "O=C(OC(C)(C)C)c1ccc(C(=O)Nc2ccc(Cl)cc2)cc1"}' http://localhost:8082/api/v1/run

Or in Python

```python

import json
import requests

def reaction_predict(reactants):
    response = requests.post(
        "http://localhost:8082/api/v1/translate",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"smiles": reactants})
    )
    return response.json()['product'][0]

product = reaction_predict('CCOCCCCO.CC(=O)Cl')
```

## Tools

- [ ] Retrosynthesis (AiZynthfinder)
- [ ] Reaction prediction (Molecular Transformer)
- [ ] Reaction fingerprints (RXNFP)


## TO-DO (missing tools)

- [ ] Condition prediction (Coley)
- [ ] LinChemIn (Syngenta)
- [ ] Reaction equilibration (Theo)
- [ ] Literature parsing (Coley, J.Cole, Jasyntho)
- [ ] Descriptors (?)
- [ ] Maybe robotic platform/simulator
- [ ] Other calculation software
- [ ] Multimodal molecule description, generation (GIT-Mol) https://github.com/AI-HPC-Research-Team/GIT-Mol/tree/main
- [ ] Multimodal too, SOTA (BioT5+) (https://huggingface.co/QizhiPei/biot5-base)
- [ ] Text to Mol (https://huggingface.co/spaces/ndhieunguyen/Lang2mol-Diff) -- checkout the app.py
- [ ] text2mol, mol2text https://github.com/QizhiPei/BioT5 (inference code. different FTs for each task?
- [ ] MolGrapher https://github.com/DS4SD/MolGrapher  image->smiles. easy as is in huggingface

saccrow-data/papers
