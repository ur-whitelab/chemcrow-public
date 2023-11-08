import os
import re
import paperqa
import langchain
import paperscraper
from langchain import SerpAPIWrapper
from pypdf.errors import PdfReadError
from langchain.tools import BaseTool
from langchain.base_language import BaseLanguageModel


def paper_search(search, pdir="query"):
    try:
        return paperscraper.search_papers(search, pdir=pdir)
    except KeyError:
        return {}

def partial(func, *args, **kwargs):
    """
    This function is a workaround for the partial function error in new langchain versions.
    This can be removed if langchain adds support for partial functions.
    """
    def wrapped(*args_wrapped, **kwargs_wrapped):
        final_args = args + args_wrapped
        final_kwargs = {**kwargs, **kwargs_wrapped}
        return func(*final_args, **final_kwargs)
    return wrapped

def scholar2result_llm(llm, query, search=None):
    """Useful to answer questions that require technical knowledge. Ask a specific question."""

    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question"],
        template="I would like to find scholarly papers to answer this question: {question}. "
        'A search query that would bring up papers that can answer this question would be: "',
    )
    query_chain = langchain.chains.LLMChain(llm=llm, prompt=prompt)

    if not os.path.isdir("./query"):
        os.mkdir("query/")

    if search is None:
        search = query_chain.run(query)
    print("\nSearch:", search)
    papers = paper_search(search, pdir=f"query/{re.sub(' ', '', search)}")

    if len(papers) == 0:
        return "Not enough papers found"
    docs = paperqa.Docs(llm=llm)
    not_loaded = 0
    for path, data in papers.items():
        try:
            docs.add(path, data["citation"])
        except (ValueError, FileNotFoundError, PdfReadError) as e:
            not_loaded += 1

    print(f"\nFound {len(papers.items())} papers but couldn't load {not_loaded}")
    return docs.query(query, length_prompt="about 100 words").answer


def web_search(keywords, search_engine="google"):
    try:
        return SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
        ).run(keywords)
    except:
        return "No results, try another search"
    

class LitSearch(BaseTool):
    name = "LiteratureSearch"
    description = (
        "Input a specific question, returns an answer from literature search. "
        "Do not mention any specific molecule names, but use more general features to formulate your questions."
    )
    llm: BaseLanguageModel
    def _run(self, query: str) -> str:
        return scholar2result_llm(self.llm, query)
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")
    
class WebSearch(BaseTool):
    name = "WebSearch"
    description = (
        "Input a specific question, returns an answer from web search. "
        "Do not mention any specific molecule names, but use more general features to formulate your questions."
    )
    def _run(self, query: str) -> str:
        return web_search(query)
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")