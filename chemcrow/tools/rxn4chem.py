from langchain.tools import BaseTool
from time import sleep
from rxn4chemistry import RXN4ChemistryWrapper
from chemcrow.utils import is_smiles

class RXNPredict(BaseTool):
    name = "RXNPredict"
    description = (
        "Predict the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )
    rxn4chem_api_key: str = ''
    rxn4chem: RXN4ChemistryWrapper = None
    base_url: str = "https://rxn.res.ibm.com"
    sleep_time: str = 5

    def __init__(self, rxn4chem_api_key):
        super(RXNPredict, self).__init__()

        self.rxn4chem_api_key = rxn4chem_api_key
        self.rxn4chem = RXN4ChemistryWrapper(
            api_key=self.rxn4chem_api_key,
            base_url=self.base_url
        )
        self.rxn4chem.create_project("ChemCrow")

    def _run(self, reactants: str) -> str:

        # Check that input is smiles
        if not is_smiles(reactants):
            return "Incorrect input."

        # Send until valid response
        while True:
            sleep(self.sleep_time)
            response = self.rxn4chem.predict_reaction(reactants)
            if "prediction_id" in response.keys():
                break

        status = "WAITING"
        while True:
            sleep(self.sleep_time)
            results = self.rxn4chem.get_predict_reaction_results(
                response["prediction_id"]
            )
            if "payload" in results["response"].keys():
                break

        res_dict = results["response"]["payload"]["attempts"][0]

        rxn = res_dict["smiles"]
        product = res_dict["productMolecule"]["smiles"]

        return product

    async def _arun(self, cas_number):
        raise NotImplementedError("Async not implemented.")
