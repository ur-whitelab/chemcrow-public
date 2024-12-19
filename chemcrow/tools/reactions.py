"""Self-hosted reaction tools. Retrosynthesis, reaction forward prediction."""

import abc
import ast
import re
from time import sleep
from typing import Optional

import requests

import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import BaseTool

from chemcrow.utils import is_smiles

__all__ = ["RXNPredictLocal", "RXNRetrosynthesisLocal"]


class RXNPredictLocal(BaseTool):
    """Predict reaction."""

    name = "ReactionPredict"
    description = (
        "Predict the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )

    def _run(self, reactants: str) -> str:
        """Run reaction prediction."""
        if not is_smiles(reactants):
            return "Incorrect input."

        product = self.predict_reaction(reactants)
        return product

    def predict_reaction(self, reactants: str) -> str:
        """Make api request."""
        try:
            response = requests.post(
                "http://localhost:8051/api/v1/run",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"smiles": reactants})
            )
            return response.json()['product'][0]
        except:
            return "Error in prediction."


class RXNRetrosynthesisLocal(BaseTool):
    """Predict retrosynthesis."""

    name = "ReactionRetrosynthesis"
    description = (
        "Obtain the synthetic route to a chemical compound. "
        "Takes as input the SMILES of the product, returns recipe."
    )
    openai_api_key: str = ""

    def _run(self, reactants: str) -> str:
        """Run reaction prediction."""
        # Check that input is smiles
        if not is_smiles(reactants):
            return "Incorrect input."

        prediction_id = self.predict_reaction(reactants)
        results = self.get_results(prediction_id)
        product = results["productMolecule"]["smiles"]
        return product

    def predict_reaction(self, reactants: str) -> str:
        """Make api request."""
        response = requests.post(
            "http://localhost:8021/api/v1/run",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"smiles": reactants})
        )
        return response.json()['product'][0]
