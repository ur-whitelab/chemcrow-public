import os
import re
from typing import Annotated

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pypdf.errors import PdfReadError
from langchain_community.utilities import SerpAPIWrapper
from .utils import is_multiple_smiles, split_smiles
from langchain_openai.embeddings import OpenAIEmbeddings
import molbloom
# import paperqa
import paperscraper


def paper_scraper(search: str,
                  pdir: str = "query",
                  semantic_scholar_api_key: str = None) -> dict:
    try:
        return paperscraper.search_papers(
            search,
            pdir=pdir,
            semantic_scholar_api_key=semantic_scholar_api_key,
        )
    except KeyError:
        return {}

def scholar2result_llm(llm,
                       query,
                       k=5,
                       max_sources=2,
                       openai_api_key=None,
                       semantic_scholar_api_key=None):
    """Useful to answer questions that require
    technical knowledge. Ask a specific question."""
    papers = paper_search(llm, query, semantic_scholar_api_key=semantic_scholar_api_key)
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

    if not_loaded > 0:
        print(f"\nFound {len(papers.items())} papers but couldn't load {not_loaded}.")
    else:
        print(f"\nFound {len(papers.items())} papers and loaded all of them.")

    answer = docs.query(query, k=k, max_sources=max_sources).formatted_answer
    return answer

def paper_search(llm, query, semantic_scholar_api_key=None):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        I would like to find scholarly papers to answer
        this question: {question}. Your response must be at
        most 10 words long.
        'A search query that would bring up papers that can answer
        this question would be: '""",
    )

    query_chain = LLMChain(llm=llm, prompt=prompt)
    if not os.path.isdir("./query"):  # todo: move to ckpt
        os.mkdir("query/")
    search = query_chain.run(query)
    print("\nSearch:", search)
    papers = paper_scraper(search, pdir=f"query/{re.sub(' ', '', search)}", semantic_scholar_api_key=semantic_scholar_api_key)
    return papers




def web_search(keywords, search_engine="google"):
    try:
        return SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
        ).run(keywords)
    except:
        return "No results, try another search"

@tool
def PatentCheck(smiles:Annotated[str,'Input SMILES']):
    '''
    Checks if compound is patented. Give this tool only one SMILES string
    '''
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

@tool
def LiteratureSearch(query:Annotated[str,'Ask a specific question'],

                     openai_api_key:Annotated[str,'openai_api_key, find it in prompt. If there is no key provided, leave this as None'],
                     semantic_scholar_api_key:Annotated[str,'semantic_scholar_api_key. If there is no key provided, leave this as None']
                     ):
    """
    LiteratureSearch
        Useful to answer questions that require technical
        knowledge. Ask a specific question.
    """
    llm=ChatOpenAI(
        api_key=openai_api_key,
        model='gpt4-0613',
        temperature=0.2,
    )
    return scholar2result_llm(
        llm,
        query,
        openai_api_key=openai_api_key,
        semantic_scholar_api_key=semantic_scholar_api_key
    )

@tool
def WebSearch(
        query:Annotated[str,'Input a specific question,Do not mention any specific molecule names, but use more general features to formulate your questions.'],
        serp_api_key:str
        ):
    '''
    WebSearch
        Input a specific question, returns an answer from web search.
        Do not mention any specific molecule names, but use more general features to formulate your questions.
    '''
    if not serp_api_key:
        return (
            "No SerpAPI key found. This tool may not be used without a SerpAPI key."
        )
    return web_search(query)


