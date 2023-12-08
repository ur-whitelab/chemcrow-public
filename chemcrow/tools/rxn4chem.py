"""Wrapper for RXN4Chem functionalities."""

import re
import ast
import abc
from typing import Optional
from langchain.tools import BaseTool
from time import sleep
from rxn4chemistry import RXN4ChemistryWrapper  # type: ignore
from chemcrow.utils import is_smiles
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


class RXN4Chem(BaseTool):
    """Wrapper for RXN4Chem functionalities."""

    name: str
    description: str
    rxn4chem_api_key: Optional[str] = None
    rxn4chem: RXN4ChemistryWrapper = None
    base_url: str = "https://rxn.res.ibm.com"
    sleep_time: int = 5

    def __init__(self, rxn4chem_api_key):
        """Init object."""
        super(RXN4Chem, self).__init__()

        self.rxn4chem_api_key = rxn4chem_api_key
        self.rxn4chem = RXN4ChemistryWrapper(
            api_key=self.rxn4chem_api_key,
            base_url=self.base_url
        )
        self.rxn4chem.project_id = '655b7b760fb57c001f25dc91'

    @abc.abstractmethod
    def _run(self, smiles: str):  # type: ignore
        """Execute operation."""
        pass

    @abc.abstractmethod
    async def _arun(self, smiles: str):
        """Async execute operation."""
        pass

    @staticmethod
    def retry(times: int, exceptions, sleep_time: int = 5):
        """
        Retry Decorator.

        Retries the wrapped function/method `times` times if the exceptions
        listed in ``exceptions`` are thrown
        :param times: The number of times to repeat the wrapped function/method
        :type times: Int
        :param Exceptions: Lists of exceptions that trigger a retry attempt
        :type Exceptions: Tuple of Exceptions
        """
        def decorator(func):
            def newfn(*args, **kwargs):
                attempt = 0
                while attempt < times:
                    try:
                        sleep(sleep_time)
                        return func(*args, **kwargs)
                    except exceptions:
                        print(
                            'Exception thrown when attempting to run %s, '
                            'attempt %d of %d' % (func, attempt, times)
                        )
                        attempt += 1
                return func(*args, **kwargs)
            return newfn
        return decorator


class RXNPredict(RXN4Chem):
    """Predict reaction."""

    name = "ReactionPredict"
    description = (
        "Predict the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )

    def _run(self, reactants: str) -> str:
        """Run reaction prediction."""
        # Check that input is smiles
        if not is_smiles(reactants):
            return "Incorrect input."

        prediction_id = self.predict_reaction(reactants)
        results = self.get_results(prediction_id)
        product = results["productMolecule"]["smiles"]
        return product

    @RXN4Chem.retry(10, KeyError)
    def predict_reaction(self, reactants: str) -> str:
        """Make api request."""
        response = self.rxn4chem.predict_reaction(reactants)
        if "prediction_id" in response.keys():
            return response["prediction_id"]
        else:
            raise KeyError

    @RXN4Chem.retry(10, KeyError)
    def get_results(self, prediction_id: str) -> str:
        """Make api request."""
        results = self.rxn4chem.get_predict_reaction_results(
            prediction_id
        )
        if "payload" in results["response"].keys():
            return results["response"]["payload"]["attempts"][0]
        else:
            raise KeyError

    async def _arun(self, cas_number):
        """Async run reaction prediction."""
        raise NotImplementedError("Async not implemented.")


class RXNRetrosynthesis(RXN4Chem):
    """Predict retrosynthesis."""

    name = "ReactionRetrosynthesis"
    description = (
        "Obtain the synthetic route to a chemical compound. "
        "Takes as input the SMILES of the product, returns recipe."
    )
    openai_api_key: str = ''

    def __init__(self, rxn4chem_api_key, openai_api_key):
        """Init object."""
        super(RXNRetrosynthesis, self).__init__(rxn4chem_api_key)
        self.openai_api_key = openai_api_key

    def _run(self, target: str) -> str:
        """Run retrosynthesis prediction."""
        # Check that input is smiles
        if not is_smiles(target):
            return "Incorrect input."

        prediction_id = self.predict_retrosynthesis(target)
        results = self.get_results(prediction_id)
        procedure = self.get_action_sequence(results)
        return procedure

    async def _arun(self, cas_number):
        """Async run retrosynthesis prediction."""
        raise NotImplementedError("Async not implemented.")

    @RXN4Chem.retry(10, KeyError)
    def predict_retrosynthesis(self, target: str) -> str:
        """Make api request."""
        response = self.rxn4chem.predict_automatic_retrosynthesis(
            product=target,
            fap=0.6,
            max_steps=3,
            nbeams=10,
            pruning_steps=2,
            ai_model="12class-tokens-2021-05-14",
        )
        if "prediction_id" in response.keys():
            return response["prediction_id"]
        raise KeyError

    @RXN4Chem.retry(20, KeyError)
    def get_results(self, prediction_id: str) -> str:
        """Make api request."""
        results = self.rxn4chem.get_predict_automatic_retrosynthesis_results(
            prediction_id
        )
        if "retrosynthetic_paths" not in results.keys():
            raise KeyError
        paths = results["retrosynthetic_paths"]
        if paths is not None:
            if len(paths) > 0:
                return paths[0]  # simply pick first path
        if results["status"] == "PROCESSING":
            sleep(self.sleep_time*2)
            raise KeyError
        print(results)
        raise KeyError

    def get_action_sequence(self, path):
        """Get sequence of actions."""
        response = self.synth_from_sequence(path["sequenceId"])
        if 'synthesis_id' not in response.keys():
            return path

        synthesis_id = response["synthesis_id"]
        nodeids = self.get_node_ids(synthesis_id)
        if nodeids is None:
            return 'Tool error'

        # Attempt to get actions for each node + product information
        real_nodes = []
        actions_and_products = []
        for node in nodeids:
            node_resp = self.get_reaction_settings(
                synthesis_id=synthesis_id, node_id=node
            )
            if 'actions' in node_resp.keys():
                real_nodes.append(node)
                actions_and_products.append(node_resp)

        json_actions = self._preproc_actions(actions_and_products)
        llm_sum = self._summary_gpt(json_actions)
        return llm_sum

    @RXN4Chem.retry(20, KeyError)
    def synth_from_sequence(self, sequence_id: str) -> str:
        """Make api request."""
        response = self.rxn4chem.create_synthesis_from_sequence(
            sequence_id=sequence_id
        )
        if "synthesis_id" in response.keys():
            return response
        raise KeyError

    @RXN4Chem.retry(20, KeyError)
    def get_node_ids(self, synthesis_id: str):
        """Make api request."""
        response = self.rxn4chem.get_node_ids(
            synthesis_id=synthesis_id
        )
        if isinstance(response, list):
            if len(response) > 0:
                return response
        return KeyError

    @RXN4Chem.retry(20, KeyError)
    def get_reaction_settings(self, synthesis_id: str, node_id: str):
        """Make api request."""
        response = self.rxn4chem.get_reaction_settings(
            synthesis_id=synthesis_id, node_id=node_id
        )
        if 'actions' in response.keys():
            return response
        elif 'response' in response.keys():
            if 'error' in response['response'].keys():
                if response['response']['error'] == 'Too Many Requests':
                    sleep(self.sleep_time*2)
                    raise KeyError
            return response
        raise KeyError

    def _preproc_actions(self, actions_and_products):
        """Preprocess actions."""
        json_actions = {'number_of_steps': len(actions_and_products)}

        for i, actn in enumerate(actions_and_products):
            json_actions[f'Step_{i}'] = {}
            json_actions[f'Step_{i}']['actions'] = actn['actions']
            json_actions[f'Step_{i}']['product'] = actn['product']

        # Clean actions to use less tokens: Remove False, None, ''
        clean_act_str = re.sub(
            r'\'[A-Za-z]+\': (None|False|\'\'),? ?',
            '',
            str(json_actions)
        )
        json_actions = ast.literal_eval(clean_act_str)

        return json_actions

    def _summary_gpt(self, json: dict) -> str:
        """Describe synthesis."""
        llm = ChatOpenAI(  # type: ignore
            temperature=0.05,
            model_name="gpt-3.5-turbo-16k",
            request_timeout=2000,
            max_tokens=2000,
            openai_api_key=self.openai_api_key
        )
        print(json)
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
