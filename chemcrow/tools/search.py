import os
import re
import paperqa
import langchain
import paperscraper
from langchain.tools import BaseTool
from langchain import SerpAPIWrapper, OpenAI
from langchain.chains import LLMChain
from pypdf.errors import PdfReadError



class LitSearch(BaseTool):
    name = "LiteratureSearch"
    description=(
        "Input a specific question, returns an answer from literature search. "
        "Do not mention any specific molecule names, but use more general features to formulate your questions."
    )
    llm: OpenAI = None
    query_chain: LLMChain = None

    def __init__(self):
        super(LitSearch, self).__init__()
        self.llm = OpenAI(
            temperature=0.05,
            model_kwargs={"stop": ['"']},
        )

        prompt = langchain.prompts.PromptTemplate(
            input_variables=["question"],
            template="I would like to find scholarly papers to answer this question: {question}. "
            'A search query that would bring up papers that can answer this question would be: "',
        )
        self.query_chain = LLMChain(llm=self.llm, prompt=prompt)

        if not os.path.isdir("./query"):
            os.mkdir("query/")

    def paper_search(self, search, pdir="query"):
        try:
            return paperscraper.search_papers(search, pdir=pdir)
        except KeyError:
            return {}

    def _run(self, query: str, search=None, npapers=16, npassages=5) -> str:
        if search is None:
            search = self.query_chain.run(query)
        print("\nSearch:", search)
        papers = self.paper_search(search, pdir=f"query/{re.sub(' ', '', search)}")

        if len(papers) == 0:
            return "Not enough papers found"

        docs = paperqa.Docs(llm=self.llm)
        not_loaded = 0
        for path, data in papers.items():
            try:
                docs.add(path, data["citation"])
            except (ValueError, FileNotFoundError, PdfReadError) as e:
                not_loaded += 1

        if not_loaded:
            print(f"\nFound {len(papers.items())} papers, couldn't load {not_loaded}")
        return docs.query(query, length_prompt="about 100 words").answer


    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class WebSearch(BaseTool):
    name = "WebSearch"
    description=(
        "Input search query, returns snippets from web search. "
        "Prefer LitSearch tool over this tool, except for simple questions."
    )
    serpapi: SerpAPIWrapper = None

    def __init__(self, search_engine='google'):
        super(WebSearch, self).__init__()

        self.serpapi = SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERP_API_KEY"),
            search_engine=search_engine
        )

    def _run(self, query: str) -> str:
        try:
            return self.serpapi.run(query)
        except:
            return "No results, try another search"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
