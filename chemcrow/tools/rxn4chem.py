import re
import ast
from langchain.tools import BaseTool
from time import sleep
from rxn4chemistry import RXN4ChemistryWrapper
from chemcrow.utils import is_smiles
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

class RXNPredict(BaseTool):
    name = "ReactionPredict"
    description = (
        "Predict the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )
    rxn4chem_api_key: str = ''
    rxn4chem: RXN4ChemistryWrapper = None
    base_url: str = "https://rxn.res.ibm.com"
    sleep_time: int = 5

    def __init__(self, rxn4chem_api_key):
        super(RXNPredict, self).__init__()

        self.rxn4chem_api_key = rxn4chem_api_key
        self.rxn4chem = RXN4ChemistryWrapper(
            api_key=self.rxn4chem_api_key,
            base_url=self.base_url
        )
        self.rxn4chem.project_id = '655b7b760fb57c001f25dc91'

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

        while True:
            sleep(self.sleep_time)
            results = self.rxn4chem.get_predict_reaction_results(
                response["prediction_id"]
            )
            if "payload" in results["response"].keys():
                break

        res_dict = results["response"]["payload"]["attempts"][0]
        product = res_dict["productMolecule"]["smiles"]

        return product

    async def _arun(self, cas_number):
        raise NotImplementedError("Async not implemented.")



class RXNPlanner(BaseTool):
    name = "ReactionPlanner"
    description = (
        "Input target molecule (SMILES), returns a sequence of reaction steps "
        "in natural language. "
    )
    rxn4chem_api_key: str = ''
    openai_api_key: str = ''
    rxn4chem: RXN4ChemistryWrapper = None
    base_url: str = "https://rxn.res.ibm.com"
    sleep_time: int = 8

    def __init__(self, rxn4chem_api_key, openai_api_key):
        super(RXNPlanner, self).__init__()

        self.rxn4chem_api_key = rxn4chem_api_key
        self.openai_api_key = openai_api_key
        self.rxn4chem = RXN4ChemistryWrapper(
            api_key=self.rxn4chem_api_key,
            base_url=self.base_url
        )
        self.rxn4chem.project_id = '655b7b760fb57c001f25dc91'

    def _run(self, target: str) -> str:

        # Check that input is smiles
        if not is_smiles(target):
            return "Incorrect input."

        else:
            # Send until valid response
            while True:
                sleep(self.sleep_time)
                response = self.rxn4chem.predict_automatic_retrosynthesis(
                    product=target,
                    fap=0.6,
                    max_steps=3,
                    nbeams=10,
                    pruning_steps=2,
                    ai_model="12class-tokens-2021-05-14",
                )
                if "prediction_id" in response.keys():
                    break

            # Get results
            while True:
                sleep(self.sleep_time)
                try:
                    print('Retrying autosynth')
                    result = self.rxn4chem.get_predict_automatic_retrosynthesis_results(
                        response["prediction_id"]
                    )
                    paths = result["retrosynthetic_paths"]
                    if paths is not None:
                        if len(paths) > 0:
                            break
                except:
                    pass

            # Simply pick first path
            selected_path = paths[0]

            if type(selected_path) == str:
                return selected_path

            # Get sequence of actions
            procedure = self._path_to_text(selected_path)
            return procedure

    async def _arun(self, target: str):
        raise NotImplementedError("Async not implemented.")

    def _path_to_text(self, path):
        for _ in range(20):
            sleep(self.sleep_time)
            print(f"\nAttempting to get synthesis from sequence: Try {_}")
            response = self.rxn4chem.create_synthesis_from_sequence(
                sequence_id=path["sequenceId"]
            )
            if "synthesis_id" in response.keys():
                break

        if 'synthesis_id' not in response.keys():
            return path

        synthesis_id = response["synthesis_id"]

        for _ in range(20):  # retry 10 times to prevent KeyError 'payload'
            print(f"\nAttempting to get nodes: Try {_}")
            sleep(self.sleep_time)
            try:
                nodeids = self.rxn4chem.get_node_ids(synthesis_id)
                print(nodeids)
                if isinstance(nodeids, list):
                    if len(nodeids)>0:
                        break
            except KeyError:
                nodeids = None
                continue
        if nodeids is None:
            return 'Tool error'

        # Attempt to get actions for each node + product information
        real_nodes = []
        actions_and_products = []
        for node in nodeids:
            while True:
                sleep(self.sleep_time)
                print(f"\nAttempting to get data for node {node}.")
                node_resp = self.rxn4chem.get_reaction_settings(
                    synthesis_id=synthesis_id, node_id=node
                )
                if 'response' in node_resp.keys():
                    # retry
                    break
                elif 'actions' in node_resp.keys():
                    real_nodes.append(node)
                    actions_and_products.append(node_resp)
                    break

        # Parse action sequences into json
        json_actions = {'number_of_steps':len(actions_and_products)}

        for i, actn in enumerate(actions_and_products):
            json_actions[f'Step_{i}'] = {}
            json_actions[f'Step_{i}']['actions'] = actn['actions']
            json_actions[f'Step_{i}']['product'] = actn['product']

        # Clean actions to use less tokens: Remove False, None, ''
        clean_act_str = re.sub(r'\'[A-Za-z]+\': (None|False|\'\'),? ?', '', str(json_actions))
        json_actions = ast.literal_eval(clean_act_str)

        llm_sum = self._summary_gpt(json_actions)

        return llm_sum  # return also smiles? to display in app

    def _summary_gpt(self, json: dict) -> str:
        """Describe synthesis."""

        llm = ChatOpenAI(
            temperature=0.05,
            model_name="gpt-3.5-turbo-16k",
            request_timeout=2000,
            max_tokens=2000,
            openai_api_key=self.openai_api_key
        )

        # Preprocess json
        for i in range(int(json['number_of_steps'])):
            step = json[f'Step_{i}']
            for s in step['actions']:
                s['messages'] = []

        prompt = (
            "Here is a chemical synthesis described as a json.\nYour task is "
            "to describe the synthesis, as if you were giving instructions for"
            "a recipe. Use only the substances, quantities, temperatures and "
            "in general any action mentioned in the json file. This is your "
            "only source of information, do not make up anything else. Also, "
            "add 15mL of DCM as a solvent in the first step. If you ever need "
            "to refer to the json file, refer to it as \"(by) the tool\". "
            "However avoid references to it. \nFor this task, give as many "
            f"details as possible.\n {str(json)}"
        )

        return llm([HumanMessage(content=prompt)]).content
