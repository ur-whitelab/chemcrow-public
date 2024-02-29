import re
import urllib
from time import sleep

import langchain
import molbloom
import pandas as pd
import pkg_resources
import requests
import tiktoken
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from langchain.tools import BaseTool

from chemcrow.utils import *
from chemcrow.utils import (
    is_multiple_smiles,
    is_smiles,
    query2cas,
    query2smiles,
    split_smiles,
    tanimoto,
)

from .prompts import safety_summary_prompt, summary_each_data


class PatentCheck(BaseTool):
    name = "PatentCheck"
    description = "Input SMILES, returns if molecule is patented. You may also input several SMILES, separated by a period."

    def _run(self, smiles: str) -> str:
        """Checks if compound is patented. Give this tool only one SMILES string"""
        if is_multiple_smiles(smiles):
            smiles_list = split_smiles(smiles)
        else:
            smiles_list = [smiles]
        try:
            output_dict = {}
            for smi in smiles_list:
                r = molbloom.buy(smi, canonicalize=True, catalog="surechembl")
                if r:
                    output_dict[smi] = "Patented"
                else:
                    output_dict[smi] = "Novel"
            return str(output_dict)
        except:
            return "Invalid SMILES string"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class MoleculeSafety:
    def __init__(self, llm: BaseLLM = None):
        while True:
            try:
                self.clintox = pd.read_csv(
                    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
                )
                break
            except (ConnectionRefusedError, urllib.error.URLError):
                sleep(5)
                continue
        self.pubchem_data = {}
        self.llm = llm

    def _fetch_pubchem_data(self, cas_number):
        """Fetch data from PubChem for a given CAS number, or use cached data if it's already been fetched."""
        if cas_number not in self.pubchem_data:
            try:
                url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/cids/JSON"
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{requests.get(url1).json()['IdentifierList']['CID'][0]}/JSON"
                r = requests.get(url)
                self.pubchem_data[cas_number] = r.json()
            except:
                return "Invalid molecule input, no Pubchem entry."
        return self.pubchem_data[cas_number]

    def ghs_classification(self, text):
        """Gives the ghs classification from Pubchem. Give this tool the name or CAS number of one molecule."""
        if is_smiles(text):
            return "Please input a valid CAS number."
        data = self._fetch_pubchem_data(text)
        if isinstance(data, str):
            return "Molecule not found in Pubchem."
        try:
            for section in data["Record"]["Section"]:
                if section.get("TOCHeading") == "Chemical Safety":
                    ghs = [
                        markup["Extra"]
                        for markup in section["Information"][0]["Value"][
                            "StringWithMarkup"
                        ][0]["Markup"]
                    ]
                    if ghs:
                        return ghs
        except (StopIteration, KeyError):
            return None

    @staticmethod
    def _scrape_pubchem(data, heading1, heading2, heading3):
        try:
            filtered_sections = []
            for section in data["Record"]["Section"]:
                toc_heading = section.get("TOCHeading")
                if toc_heading == heading1:
                    for section2 in section["Section"]:
                        if section2.get("TOCHeading") == heading2:
                            for section3 in section2["Section"]:
                                if section3.get("TOCHeading") == heading3:
                                    filtered_sections.append(section3)
            return filtered_sections
        except:
            return None

    def _get_safety_data(self, cas):
        data = self._fetch_pubchem_data(cas)
        safety_data = []

        iterations = [
            (
                [
                    "Health Hazards",
                    "GHS Classification",
                    "Hazards Summary",
                    "NFPA Hazard Classification",
                ],
                "Safety and Hazards",
                "Hazards Identification",
            ),
            (
                ["Explosive Limits and Potential", "Preventive Measures"],
                "Safety and Hazards",
                "Safety and Hazard Properties",
            ),
            (
                [
                    "Inhalation Risk",
                    "Effects of Long Term Exposure",
                    "Personal Protective Equipment (PPE)",
                ],
                "Safety and Hazards",
                "Exposure Control and Personal Protection",
            ),
            (
                ["Toxicity Summary", "Carcinogen Classification"],
                "Toxicity",
                "Toxicological Information",
            ),
        ]

        for items, header1, header2 in iterations:
            safety_data.extend(
                [self._scrape_pubchem(data, header1, header2, item)] for item in items
            )

        return safety_data

    @staticmethod
    def _num_tokens(string, encoding_name="text-davinci-003"):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_safety_summary(self, cas):
        safety_data = self._get_safety_data(cas)
        approx_length = int(
            (3500 * 4) / len(safety_data) - 0.1 * ((3500 * 4) / len(safety_data))
        )
        prompt_short = PromptTemplate(
            template=summary_each_data, input_variables=["data", "approx_length"]
        )
        llm_chain_short = LLMChain(prompt=prompt_short, llm=self.llm)

        llm_output = []
        for info in safety_data:
            if self._num_tokens(str(info)) > approx_length:
                trunc_info = str(info)[:approx_length]
                llm_output.append(
                    llm_chain_short.run(
                        {"data": str(trunc_info), "approx_length": approx_length}
                    )
                )
            else:
                llm_output.append(
                    llm_chain_short.run(
                        {"data": str(info), "approx_length": approx_length}
                    )
                )
        return llm_output


class SafetySummary(BaseTool):
    name = "SafetySummary"
    description = (
        "Input CAS number, returns a summary of safety information."
        "The summary includes Operator safety, GHS information, "
        "Environmental risks, and Societal impact."
    )
    llm: BaseLLM = None
    llm_chain: LLMChain = None
    pubchem_data: dict = dict()
    mol_safety: MoleculeSafety = None

    def __init__(self, llm):
        super().__init__()
        self.mol_safety = MoleculeSafety(llm=llm)
        self.llm = llm
        prompt = PromptTemplate(
            template=safety_summary_prompt, input_variables=["data"]
        )
        self.llm_chain = LLMChain(prompt=prompt, llm=self.llm)

    def _run(self, cas: str) -> str:
        if is_smiles(cas):
            return "Please input a valid CAS number."
        data = self.mol_safety._fetch_pubchem_data(cas)
        if isinstance(data, str):
            return "Molecule not found in Pubchem."

        data = self.mol_safety.get_safety_summary(cas)
        return self.llm_chain.run(" ".join(data))

    async def _arun(self, cas_number):
        raise NotImplementedError("Async not implemented.")


class ExplosiveCheck(BaseTool):
    name = "ExplosiveCheck"
    description = "Input CAS number, returns if molecule is explosive."
    mol_safety: MoleculeSafety = None

    def __init__(self):
        super().__init__()
        self.mol_safety = MoleculeSafety()

    def _run(self, cas_number):
        """Checks if a molecule has an explosive GHS classification using pubchem."""
        # first check if the input is a CAS number
        if is_smiles(cas_number):
            return "Please input a valid CAS number."
        cls = self.mol_safety.ghs_classification(cas_number)
        if cls is None:
            return (
                "Explosive Check Error. The molecule may not be assigned a GHS rating. "
            )
        if "Explos" in str(cls) or "explos" in str(cls):
            return "Molecule is explosive"
        else:
            return "Molecule is not known to be explosive"

    async def _arun(self, cas_number):
        raise NotImplementedError("Async not implemented.")


class SimilarControlChemCheck(BaseTool):
    name = "SimilarityToControlChem"
    description = "Input SMILES, returns similarity to controlled chemicals."

    def _run(self, smiles: str) -> str:
        """Checks max similarity between compound and controlled chemicals.
        Input SMILES string."""

        data_path = pkg_resources.resource_filename("chemcrow", "data/chem_wep_smi.csv")
        cw_df = pd.read_csv(data_path)

        try:
            if not is_smiles(smiles):
                return "Please input a valid SMILES string."

            max_sim = (
                cw_df["smiles"]
                .apply(lambda x: self.tanimoto(smiles, x))
                .max()
            )
            if max_sim > 0.35:
                return (
                    f"{smiles} has a high similarity "
                    f"({max_sim:.4}) to a known controlled chemical."
                )
            else:
                return (
                    f"{smiles} has a low similarity "
                    f"({max_sim:.4}) to a known controlled chemical."
                    "This is substance is safe, you may proceed with the task."
                )
        except:
            return "Tool error."

    def tanimoto(self, s1, s2):
        sim = tanimoto(s1,s2)
        if isinstance(sim, float):
            return sim
        return 0.0

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class ControlChemCheck(BaseTool):
    name = "ControlChemCheck"
    description = "Input CAS number, True if molecule is a controlled chemical."
    similar_control_chem_check = SimilarControlChemCheck()

    def _run(self, query: str) -> str:
        """Checks if compound is a controlled chemical. Input CAS number."""
        data_path = pkg_resources.resource_filename("chemcrow", "data/chem_wep_smi.csv")
        cw_df = pd.read_csv(data_path)
        try:
            if is_smiles(query):
                query_esc = re.escape(query)
                found = (
                    cw_df["smiles"]
                    .astype(str)
                    .str.contains(f"^{query_esc}$", regex=True)
                    .any()
                )
            else:
                found = (
                    cw_df["cas"]
                    .astype(str)
                    .str.contains(f"^\({query}\)$", regex=True)
                    .any()
                )
            if found:
                return (
                    f"The molecule {query} appears in a list of "
                    "controlled chemicals."
                )
            else:
                # Get smiles of CAS number
                try:
                    smi = query2smiles(query)
                except ValueError as e:
                    return str(e)
                # Check similarity to known controlled chemicals
                return self.similar_control_chem_check._run(smi)

        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class Query2SMILES(BaseTool):
    name = "Name2SMILES"
    description = "Input a molecule name, returns SMILES."
    url: str = None
    ControlChemCheck = ControlChemCheck()

    def __init__(
        self,
    ):
        super().__init__()
        self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"

    def _run(self, query: str) -> str:
        """This function queries the given molecule name and returns a SMILES string from the record"""
        """Useful to get the SMILES string of one molecule by searching the name of a molecule. Only query with one specific name."""
        try:
            smi = query2smiles(query, self.url)
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


class Query2CAS(BaseTool):
    name = "Mol2CAS"
    description = "Input molecule (name or SMILES), returns CAS number."
    url_cid: str = None
    url_data: str = None
    ControlChemCheck = ControlChemCheck()

    def __init__(
        self,
    ):
        super().__init__()
        self.url_cid = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON"
        )
        self.url_data = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
        )

    def _run(self, query: str) -> str:
        try:
            # if query is smiles
            smiles = None
            if is_smiles(query):
                smiles = query
            try:
                cas = query2cas(query, self.url_cid, self.url_data)
            except ValueError as e:
                return str(e)
            if smiles is None:
                try:
                    smiles = query2smiles(cas, None)
                except ValueError as e:
                    return str(e)
            # great now check if smiles is controlled
            msg = self.ControlChemCheck._run(smiles)
            if "high similarity" in msg or "appears" in msg:
                return f"CAS number {cas}found, but " + msg
            return cas
            # check if smiles is controlled
        except ValueError:
            return "CAS number not found"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
