
# Tools of organic chemistry

A docker container was prepared for each tool, which exposes an api for requests.

> docker run -d -p 8052:5000 doncamilom/rxnpred:latest

Where 5000 is fixed, and 8082 is the port to be exposed.

A request in curl can look like this

> curl -X POST -H "Content-Type: application/json" -d '{"smiles": "O=C(OC(C)(C)C)c1ccc(C(=O)Nc2ccc(Cl)cc2)cc1"}' http://localhost:8082/api/v1/run

Or in Python

```python

import json
import requests

def reaction_predict(reactants):
    response = requests.post(
        "http://localhost:8052/api/v1/run",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"smiles": reactants})
    )
    return response.json()['product'][0]

product = reaction_predict('CCOCCCCO.CC(=O)Cl')
```
