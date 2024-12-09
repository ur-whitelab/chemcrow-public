
from typing import Type

import molbloom
import pandas as pd
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from chemcrow.utils import is_smiles


class ChemSpace:
    def __init__(self, chemspace_api_key=None):
        self.chemspace_api_key = chemspace_api_key
        self._renew_token()  # Create token

    def _renew_token(self):
        self.chemspace_token = requests.get(
            url="https://api.chem-space.com/auth/token",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.chemspace_api_key}",
            },
        ).json()["access_token"]

    def _make_api_request(
        self,
        query,
        request_type,
        count,
        categories,
    ):
        """
        Make a generic request to chem-space API.

        Categories request.
            CSCS: Custom Request: Could be useful for requesting whole synthesis
            CSMB: Make-On-Demand Building Blocks
            CSSB: In-Stock Building Blocks
            CSSS: In-stock Screening Compounds
            CSMS: Make-On-Demand Screening Compounds
        """

        def _do_request():
            data = requests.request(
                "POST",
                url=f"https://api.chem-space.com/v3/search/{request_type}?count={count}&page=1&categories={categories}",
                headers={
                    "Accept": "application/json; version=3.1",
                    "Authorization": f"Bearer {self.chemspace_token}",
                },
                data={"SMILES": f"{query}"},
            ).json()
            return data

        data = _do_request()

        # renew token if token is invalid
        if "message" in data.keys():
            if data["message"] == "Your request was made with invalid credentials.":
                self._renew_token()

        data = _do_request()
        return data

    def _convert_single(self, query, search_type: str):
        """Do query for a single molecule"""
        data = self._make_api_request(query, "exact", 1, "CSCS,CSMB,CSSB")
        if data["count"] > 0:
            return data["items"][0][search_type]
        else:
            return "No data was found for this compound."

    def convert_mol_rep(self, query, search_type: str = "smiles"):
        if ", " in query:
            query_list = query.split(", ")
        else:
            query_list = [query]
        smi = ""
        try:
            for q in query_list:
                smi += f"{query}'s {search_type} is: {str(self._convert_single(q, search_type))}"
                return smi
        except Exception:
            return "The input provided is wrong. Input either a single molecule, or multiple molecules separated by a ', '"

    def buy_mol(
        self,
        smiles,
        request_type="exact",
        count=1,
    ):
        """
        Get data about purchasing compounds.

        smiles: smiles string of the molecule you want to buy
        request_type: one of "exact", "sim" (search by similarity), "sub" (search by substructure).
        count: retrieve data for this many substances max.
        """

        def purchasable_check(
            s,
        ):
            if not is_smiles(s):
                try:
                    s = self.convert_mol_rep(s, "smiles")
                except:
                    return "Invalid SMILES string."

            """Checks if molecule is available for purchase (ZINC20)"""
            try:
                r = molbloom.buy(s, canonicalize=True)
            except:
                print("invalid smiles")
                return False
            if r:
                return True
            else:
                return False

        purchasable = purchasable_check(smiles)

        if request_type == "exact":
            categories = "CSMB,CSSB"
        elif request_type in ["sim", "sub"]:
            categories = "CSSS,CSMS"

        data = self._make_api_request(smiles, request_type, count, categories)

        try:
            if data["count"] == 0:
                if purchasable:
                    return "Compound is purchasable, but price is unknown."
                else:
                    return "Compound is not purchasable."
        except KeyError:
            return "Invalid query, try something else. "

        print(f"Obtaining data for {data['count']} substances.")

        dfs = []
        # Convert this data into df
        for item in data["items"]:
            dfs_tmp = []
            smiles = item["smiles"]
            offers = item["offers"]

            for off in offers:
                df_tmp = pd.DataFrame(off["prices"])
                df_tmp["vendorName"] = off["vendorName"]
                df_tmp["time"] = off["shipsWithin"]
                df_tmp["purity"] = off["purity"]

                dfs_tmp.append(df_tmp)

            df_this = pd.concat(dfs_tmp)
            df_this["smiles"] = smiles
            dfs.append(df_this)

        df = pd.concat(dfs).reset_index(drop=True)

        df["quantity"] = df["pack"].astype(str) + df["uom"]
        df["time"] = df["time"].astype(str) + " days"

        df = df.drop(columns=["pack", "uom"])
        # Remove all entries that are not numbers
        df = df[df["priceUsd"].astype(str).str.isnumeric()]

        cheapest = df.iloc[df["priceUsd"].astype(float).idxmin()]
        return f"{cheapest['quantity']} of this molecule cost {cheapest['priceUsd']} USD and can be purchased at {cheapest['vendorName']}."


class GetMoleculePrice(BaseTool):
    name: str = Field(default="GetMoleculePrice")
    description: str = Field(default="Get the cheapest available price of a molecule.")
    chemspace_api_key: str = Field(default=None)
    url: str = Field(default=None)
    args_schema: Type[BaseModel] = Field(
        default=None,
        exclude=True
    )
    class InputSchema(BaseModel):
        query: str = Field(
            description="SMILES string or chemical name of the molecule to look up"
        )
    

    def __init__(self, chemspace_api_key: str = None):
        super().__init__()
        self.chemspace_api_key = chemspace_api_key
        self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"

    def _run(self, query: str) -> str:
        if not self.chemspace_api_key:
            return "No Chemspace API key found. This tool may not be used without a Chemspace API key."
        try:
            chemspace = ChemSpace(self.chemspace_api_key)
            price = chemspace.buy_mol(query)
            return price
        except Exception as e:
            return str(e)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
