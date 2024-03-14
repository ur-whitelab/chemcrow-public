import os
import re

import langchain
import molbloom
import paperqa
import paperscraper
from langchain import SerpAPIWrapper
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.embeddings.openai import OpenAIEmbeddings
from pypdf.errors import PdfReadError

from chemcrow.utils import is_multiple_smiles, split_smiles


def paper_scraper(search: str, pdir: str = "query") -> dict:
    try:
        return paperscraper.search_papers(search, pdir=pdir)
    except KeyError:
        return {}


def paper_search(llm, query):
    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question"],
        template="""
        I would like to find scholarly papers to answer
        this question: {question}. Your response must be at
        most 10 words long.
        'A search query that would bring up papers that can answer
        this question would be: '""",
    )

    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)
    if not os.path.isdir("./query"):  # todo: move to ckpt
        os.mkdir("query/")
    search = query_chain.run(query)
    print("\nSearch:", search)
    papers = paper_scraper(search, pdir=f"query/{re.sub(' ', '', search)}")
    return papers


def scholar2result_llm(llm, query, k=5, max_sources=2, openai_api_key=None):
    """Useful to answer questions that require
    technical knowledge. Ask a specific question."""
    papers = paper_search(llm, query)
    if len(papers) == 0:
        return "Not enough papers found"
    docs = paperqa.Docs(
        llm=llm,
        summary_llm=llm,
        embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    not_loaded = 0
    for path, data in papers.items():
        try:
            docs.add(path, data["citation"])
        except (ValueError, FileNotFoundError, PdfReadError):
            not_loaded += 1

    print(f"\nFound {len(papers.items())} papers but couldn't load {not_loaded}")
    answer = docs.query(query, k=k, max_sources=max_sources).formatted_answer
    return answer


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = (
        "Useful to answer questions that require technical "
        "knowledge. Ask a specific question."
    )
    llm: BaseLanguageModel = None
    api_key: str = None

    def __init__(self, llm, api_key):
        super().__init__()
        self.llm = llm
        self.api_key = api_key

    def _run(self, query) -> str:
        return scholar2result_llm(self.llm, query, openai_api_key=self.api_key)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")


def web_search(keywords, search_engine="google"):
    try:
        return SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
        ).run(keywords)
    except:
        return "No results, try another search"


class WebSearch(BaseTool):
    name = "WebSearch"
    description = (
        "Input a specific question, returns an answer from web search. "
        "Do not mention any specific molecule names, but use more general features to formulate your questions."
    )
    serp_api_key: str = None

    def __init__(self, serp_api_key: str = None):
        super().__init__()
        self.serp_api_key = serp_api_key

    def _run(self, query: str) -> str:
        if not self.serp_api_key:
            return (
                "No SerpAPI key found. This tool may not be used without a SerpAPI key."
            )
        return web_search(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")


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
