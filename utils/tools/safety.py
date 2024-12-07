import re
import urllib
from time import sleep

import langchain
import molbloom
import pandas as pd
import pkg_resources
import requests
import tiktoken
from langchain_core.prompts import PromptTemplate
from langchain.llms import BaseLLM
from langchain.chains import LLMChain
from langchain.tools import BaseTool

from .utils import is_smiles, pubchem_query2smiles, tanimoto

from .prompts import safety_summary_prompt, summary_each_data

from langchain_core.tools import tool
from typing import Annotated
from .make_llm import sample_llm,make_llm

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


@tool
def ControlChemCheck(query:Annotated[str,'Input CAS number']):
    '''
    ControlChemCheck
        Input CAS number, True if molecule is a controlled chemical.

    '''
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
                smi = pubchem_query2smiles(query)
            except ValueError as e:
                return str(e)
            # Check similarity to known controlled chemicals
            return SimilarControlChemCheck(smi)

    except Exception as e:
        return f"Error: {e}"

@tool
def SimilarControlChemCheck(smiles:Annotated[str,'Input SMILES string.']):
    """
    SimilarControlChemCheck
        Input SMILES, returns similarity to controlled chemicals
        Checks max similarity between compound and controlled chemicals

    """

    def _tanimoto(s1, s2):
        sim = tanimoto(s1, s2)
        if isinstance(sim, float):
            return sim
        return 0.0

    data_path = pkg_resources.resource_filename("chemcrow", "data/chem_wep_smi.csv")
    cw_df = pd.read_csv(data_path)
    try:
        if not is_smiles(smiles):
            return "Please input a valid SMILES string."

        max_sim = cw_df["smiles"].apply(lambda x: _tanimoto(smiles, x)).max()
        if max_sim > 0.35:
            return (
                f"{smiles} has a high similarity "
                f"({max_sim:.4}) to a known controlled chemical."
            )
        else:
            return (
                f"{smiles} has a low similarity "
                f"({max_sim:.4}) to a known controlled chemical. "
                "This is substance is safe, you may proceed with the task."
            )
    except:
        return "Tool error."

@tool
def ExplosiveCheck(cas_number:Annotated[str,'Input CAS number'],llm=sample_llm):
    """
    ExplosiveCheck
        Input CAS number, returns if molecule is explosive.
        Checks if a molecule has an explosive GHS classification using pubchem.
    """
    mol_safety = MoleculeSafety(llm=llm)
    if is_smiles(cas_number):
        return "Please input a valid CAS number."
    cls = mol_safety.ghs_classification(cas_number)
    if cls is None:
        return (
            "Explosive Check Error. The molecule may not be assigned a GHS rating. "
        )
    if "Explos" in str(cls) or "explos" in str(cls):
        return "Molecule is explosive"
    else:
        return "Molecule is not known to be explosive"

@tool
def SafetySummary(cas: Annotated[str,'Input CAS number']):
    """
    SafetySummary
        Checks if a molecule has an explosive GHS classification using pubchem.
        Input CAS number, returns a summary of safety information.
        The summary includes Operator safety, GHS information,Environmental risks, and Societal impact.

    """

    llm=sample_llm #

    mol_safety = MoleculeSafety(llm=llm)
    prompt = PromptTemplate(
        template=safety_summary_prompt, input_variables=["data"]
    )
    llm_chain=LLMChain(prompt=prompt, llm=llm)
    if is_smiles(cas):
        return "Please input a valid CAS number."
    data = mol_safety._fetch_pubchem_data(cas)
    if isinstance(data, str):
        return "Molecule not found in Pubchem."

    data = mol_safety.get_safety_summary(cas)
    return llm_chain.run(" ".join(data))