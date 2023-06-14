import os
import re
import paperqa
import langchain
import paperscraper
from langchain.tools import BaseTool
from langchain import SerpAPIWrapper, OpenAI
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from pypdf.errors import PdfReadError
from pydantic import validator
from typing import Optional
from datetime import datetime


class LitSearch(BaseTool):
    name = "LiteratureSearch"
    description = (
        "Input a specific question, returns an answer from literature search. "
        "Do not mention any specific molecule names, but use more general features to formulate your questions."
    )
    llm: BaseLanguageModel
    query_chain: Optional[LLMChain] = None
    pdir: str = "query"
    searches: int = 2
    verobse: bool = False
    docs: Optional[paperqa.Docs] = None

    @validator("query_chain", always=True)
    def init_query_chain(cls, v, values):
        if v is None:
            search_prompt = langchain.prompts.PromptTemplate(
                input_variables=["question", "count"],
                template="We want to answer the following question: {question} \n"
                "Provide {count} keyword searches (one search per line) "
                "that will find papers to help answer the question. "
                "Do not use boolean operators. "
                "Make some searches broad and some narrow. "
                "Do not use boolean operators or quotes.\n\n"
                "1. ",
            )
            v = LLMChain(llm=values["llm"], prompt=search_prompt)
        return v

    @validator("pdir", always=True)
    def init_pdir(cls, v):
        if not os.path.isdir(v):
            os.mkdir(v)
        return v

    def paper_search(self, search):
        try:
            return paperscraper.search_papers(
                search, pdir=self.pdir, batch_size=6, limit=4, verbose=False
            )
        except KeyError:
            return {}

    def _run(self, query: str) -> str:

        if self.verbose:
            print("\n\nChoosing search terms\n1. ", end="")
        searches = self.query_chain.run(question=query, count=self.searches)
        print("")
        queries = [s for s in searches.split("\n") if len(s) > 3]
        # remove 2., 3. from queries
        queries = [re.sub(r"^\d+\.\s*", "", q) for q in queries]
        # remove quotes
        queries = [re.sub(r"\"", "", q) for q in queries]
        papers = {}
        for q in queries:
            papers.update(self.paper_search(q))
            if self.verbose:
                print(f"retrieved {len(papers)} papers total")

        if len(papers) == 0:
            return "Not enough papers found"
        if self.docs is None:
            self.docs = paperqa.Docs(
                llm=self.llm, summary_llm="gpt-3.5-turbo", memory=True
            )
        not_loaded = 0
        for path, data in papers.items():
            try:
                self.docs.add(path, citation=data["citation"], docname=data["key"])
            except (ValueError, PdfReadError) as e:
                not_loaded += 1

        if not_loaded:
            print(f"\nFound {len(papers.items())} papers, couldn't load {not_loaded}")
        return self.docs.query(query, length_prompt="about 100 words").answer

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class WebSearch(BaseTool):
    name = "WebSearch"
    description = (
        "Input search query, returns snippets from web search. "
        "Prefer LitSearch tool over this tool, except for simple questions."
    )
    serpapi: SerpAPIWrapper = None

    def __init__(self, search_engine="google"):
        super(WebSearch, self).__init__()

        self.serpapi = SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
        )

    def _run(self, query: str) -> str:
        try:
            return self.serpapi.run(query)
        except:
            return "No results, try another search"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
