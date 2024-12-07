from streamlit import chat_message

from utils.agents import ChemCrewAgent,get_messages_input
from langgraph.graph.graph import START,END

from utils.tools.make_llm import make_llm
from langgraph.graph.message import MessagesState

# from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.state import StateGraph


from operator import add
from typing import TypedDict, List, Tuple, Annotated

import streamlit as st

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [('ai','how can I assist you today')]

class SampleState(TypedDict):
    messages: Annotated[List[Tuple[str, str]],add]

class MyGraph:
    _llm,_tool_agent,_tools=ChemCrewAgent(
            model="gpt-4-0613",
            temp=0.1,
            openai_api_key='openai_api_key'
        )
    @staticmethod
    def show_chat_history(state: SampleState) -> SampleState:
        for role, content in state['messages']:
            st.chat_message(role).write(content)
        return {'messages': list()}

    @classmethod
    def call_tool_agent(cls,state: SampleState) -> SampleState:
        chain = cls._llm | StrOutputParser()
        role,_query=state['messages'][-1]
        msgs=get_messages_input(query=_query)
        tool_rt = cls._tool_agent.invoke(msgs)['messages'][-1]
        prompts = [
            ('system', 'rephrase the content from tool'),
            ('human', f'tool:[{tool_rt}]')
        ]

        rt = st.chat_message('ai').write_stream(chain.stream(prompts))
        return {'messages': [('ai', rt)]}

    def __new__(cls):
        _build= StateGraph(SampleState)
        _build.add_node('show_chat_history', cls.show_chat_history)
        _build.add_node('call_tool_agent',cls.call_tool_agent)

        _build.add_edge(START,'show_chat_history')
        _build.add_edge('show_chat_history','call_tool_agent')
        _build.add_edge('call_tool_agent',END)

        graph=_build.compile()
        return graph

if __name__ == '__main__':

    st.sidebar.write(
        st.session_state.chat_history
    )

    Graph = MyGraph()
    state=SampleState()
    human=st.chat_input('What is the molecular weight of tylenol?')
    if human:
        st.session_state.chat_history.append(('human',human))
        state['messages']=st.session_state.chat_history
        state=Graph.invoke(state)
        st.session_state.chat_history=state['messages']




