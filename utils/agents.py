from docutils.parsers.rst.directives.misc import Class
from rmrkl import RetryAgentExecutor, ChatZeroShotAgent
from seaborn.external.appdirs import system

from .tools.make_llm import make_llm
from .tools.make_tools import make_tools
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .agent_prompts import *

class ChemCrewAgent():
    @classmethod
    def create_default_tool(
            cls,
            tools,
            openai_api_key,
            api_keys: dict,
            tools_model="gpt-4-0613",
            temp=0.1,

    ):

        _llm = make_llm(
            model=tools_model,
            api_key=openai_api_key,
            temp=temp,
        )

        tools = make_tools(_llm,api_keys=api_keys)+tools


        return create_react_agent(_llm,tools),tools


    def __new__(
            cls,
            openai_api_key,
            api_keys : dict= {},
            tools:list=None,
            model="gpt-4-0613",
            tools_model="gpt-4-0613",
            temp=0.1,
            max_iterations=40
    ):
        llm=make_llm(
            model=model,
            temp=temp,
            api_key=openai_api_key
        )
        if tools is None:
            tools=list()

        tool_agent,tools=cls.create_default_tool(
            openai_api_key=openai_api_key,
            tools=tools,
            tools_model=tools_model,
            temp=temp,
            api_keys=api_keys
        )

        return llm, tool_agent,tools

def get_messages_input(query:str= 'What is the molecular weight of tylenol?'):
    _,_,tools=ChemCrewAgent(
        model="gpt-4-0613",
        temp=0.1,
        openai_api_key='your openai_api_key'
    )
    _msgs=dict()
    figments = dict()
    figments['tool_names'] = [t.name for t in tools]
    figments['input'] = query
    _msgs['messages'] = chat_prompt.invoke(figments).to_messages()
    return _msgs


if __name__ == '__main__':
    _llm,_tool_agent,_tools=ChemCrewAgent(
        model="gpt-4-0613",
        temp=0.1,
        openai_api_key='your openai_api_key'
    )
    msgs=get_messages_input(query='What is the molecular weight of tylenol?')
    print(_tool_agent.invoke(msgs)['messages'][-1])

