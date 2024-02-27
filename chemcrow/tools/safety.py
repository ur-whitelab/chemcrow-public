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
from rdkit import Chem

from chemcrow.utils import *
from chemcrow.utils import is_cas, is_smiles, tanimoto

from .prompts import safety_summary_prompt, summary_each_data


def query2smiles(
    query: str,
    url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}",
) -> str:
    if url is None:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    r = requests.get(url.format(query, "property/IsomericSMILES/JSON"))
    # convert the response to a json object
    data = r.json()
    # return the SMILES string
    try:
        smi = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except KeyError:
        return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, input one molecule at a time."
    return str(Chem.CanonSmiles(largest_mol(smi)))


def query2cas(query: str, url_cid: str, url_data: str):
    try:
        mode = "name"
        if is_smiles(query):
            mode = "smiles"
        url_cid = url_cid.format(mode, query)
        cid = requests.get(url_cid).json()["IdentifierList"]["CID"][0]
        url_data = url_data.format(cid)
        data = requests.get(url_data).json()
    except (requests.exceptions.RequestException, KeyError):
        raise ValueError("Invalid molecule input, no Pubchem entry")

    try:
        for section in data["Record"]["Section"]:
            if section.get("TOCHeading") == "Names and Identifiers":
                for subsection in section["Section"]:
                    if subsection.get("TOCHeading") == "Other Identifiers":
                        for subsubsection in subsection["Section"]:
                            if subsubsection.get("TOCHeading") == "CAS":
                                return subsubsection["Information"][0]["Value"][
                                    "StringWithMarkup"
                                ][0]["String"]
    except KeyError:
        raise ValueError("Invalid molecule input, no Pubchem entry")

    raise ValueError("CAS number not found")


class PatentCheck(BaseTool):
    name = "PatentCheck"
    description = "Input SMILES, returns if molecule is patented"

    def _run(self, smiles: str) -> str:
        """Checks if compound is patented. Give this tool only one SMILES string"""
        try:
            r = molbloom.buy(smiles, canonicalize=True, catalog="surechembl")
        except:
            return "Invalid SMILES string"
        if r:
            return "Patented"
        else:
            return "Novel"

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

    def safety_summary(self, cas):
        if is_smiles(cas):
            return "Please input a valid CAS number."
        data = self._fetch_pubchem_data(cas)
        if isinstance(data, str):
            return "Molecule not found in Pubchem."

        data = self.get_safety_summary(cas)

        prompt = PromptTemplate(template=prompt_template, input_variables=["data"])
        llm_chain = LLMChain(prompt=safety_summary_prompt, llm=self.llm)
        return llm_chain.run(" ".join(data))


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
            return "Molecule is not known to be explosive."

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
                .apply(lambda x: tanimoto(smiles, x))
                .replace("Error: Not a valid SMILES string", 0.0)
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
                )
        except:
            return "Tool error."

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class ControlChemCheck(BaseTool):
    name = "ControlChemCheck"
    description = "Input CAS number, True if molecule is a controlled chemical."
    similar_control_chem_check = SimilarControlChemCheck()

    def _run(self, query: str) -> str:
        """Checks if compound is known controlled chemical. Input CAS number."""
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
                    f"The CAS number {query} appears in a list of "
                    "controlled chemicals."
                )
            else:
                # Get smiles of CAS number
                smi = query2smiles(query)
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
        smi = query2smiles(query, self.url)
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
            cas = query2cas(query, self.url_cid, self.url_data)
            if smiles is None:
                smiles = query2smiles(query, None)
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
