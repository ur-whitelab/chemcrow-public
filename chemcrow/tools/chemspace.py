import requests
import pandas as pd
import molbloom
import os
from langchain.tools import BaseTool

from chemcrow.utils import is_smiles, pubchem_query2smiles
from chemcrow.tools.safety import ControlChemCheck

class ChemSpace:
    def __init__(self, chemspace_key=None):
        self.chemspace_api_key = chemspace_key
        self._set_chemspace_api_key()
        self._renew_token()  # Create token

    def _renew_token(self):
        self.chemspace_token = requests.get(
            url="https://api.chem-space.com/auth/token",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.chemspace_api_key}",
            },
        ).json()["access_token"]

    def _set_chemspace_api_key(self)->None:
        if not self.chemspace_api_key:
            # see if there is an api key in os.getenv("CHEMSPACE_API_KEY") boolean
            api_key = os.getenv("CHEMSPACE_API_KEY")
            if not api_key:
                raise ValueError(
                    "No ChemSpace API key found. Please set it in the environment variable 'CHEMSPACE_API_KEY'."
                )
        else: 
            os.environ["CHEMSPACE_API_KEY"] = self.chemspace_api_key
        return None


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

    def query2smiles(self, query):
        """
        Get smiles string for some molecule.
        Input can be common name, iupac, etc.
        Also handles query with multiple molecules separated by ", "
        """

        def _q2s_single(query):
            """Do query for a single molecule"""
            data = self._make_api_request(query, "exact", 1, "CSCS,CSMB,CSSB")
            if data["count"] > 0:
                return data["items"][0]["smiles"]
            else:
                return "No data was found for this compound."

        if ", " in query:
            query_list = query.split(", ")
        else:
            query_list = [query]

        try:
            smi_list = list(map(_q2s_single, query_list))
            return ".".join(smi_list)
        except:
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
                s = self.query2smiles(s)

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
        df = df[df['priceUsd'].astype(str).str.isnumeric()]

        cheapest = df.iloc[df["priceUsd"].astype(float).idxmin()]
        return f"{cheapest['quantity']} of this molecule cost {cheapest['priceUsd']} USD and can be purchased at {cheapest['vendorName']}."
    

class Query2SMILES(BaseTool):
    name = "Name2SMILES"
    description = "Input a molecule name, returns SMILES."
    url: str = None
    chemspace_api_key: str = None
    ControlChemCheck = ControlChemCheck()

    def __init__(
        self, chemspace_api_key: str = None
    ):
        super().__init__()
        self.chemspace_api_key = chemspace_api_key
        self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"

    def _run(self, query: str) -> str:
        """This function queries the given molecule name and returns a SMILES string from the record"""
        """Useful to get the SMILES string of one molecule by searching the name of a molecule. Only query with one specific name."""
        #first see if api key is set
        try:
            chemspace = ChemSpace(self.chemspace_api_key)
            smi = chemspace.query2smiles(query)
        except Exception as e:
            try:
                smi = pubchem_query2smiles(query, self.url)
            except ValueError as e:
                return str(e)
        # check if smiles is controlled
        msg = "Note: " + self.ControlChemCheck._run(smi)
        if "high similarity" in msg or "appears" in msg:
            return f"CAS number {smi}found, but " + msg
        return smi

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()