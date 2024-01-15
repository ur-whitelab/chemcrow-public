import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from IPython.core.display import HTML
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from PIL import Image
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from chemcrow.agents import ChemCrow

# from chemcrow
from chemcrow.agents.prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, SUFFIX
from chemcrow.frontend.streamlit_callback_handler import StreamlitCallbackHandlerChem
from chemcrow.frontend.utils import cdk

# from chemcrow.mol_utils.generals import is_smiles


load_dotenv()
ss = st.session_state

# tools = ChemTools().all_tools


agent = ChemCrow(
    # tools,
    model="gpt-4",
    temp=0.1,
).agent_executor


# tool_list = pd.Series(
#    {f"✅ {t.name}":t.description for t in tools}
# ).reset_index()
# tool_list.columns = ['Tool', 'Description']


icon = Image.open("assets/logo0.png")
st.set_page_config(page_title="ChemCrow", page_icon=icon)

# Set width of sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 450px;
        max-width: 450px;
    }
    """,
    unsafe_allow_html=True,
)


# Session state
# st.session_state['molecule'] = "CCO"
def on_api_key_change():
    api_key = ss.get("api_key") or os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key


# sidebar
with st.sidebar:
    chemcrow_logo = Image.open("assets/chemcrow-logo-bold-new.png")
    st.image(chemcrow_logo)

    # Input OpenAI api key
    st.markdown("Input your OpenAI API key.")
    st.text_input(
        "OpenAI API key",
        type="password",
        key="api_key",
        on_change=on_api_key_change,
        label_visibility="collapsed",
    )

    # Display available tools
    # st.markdown(f"# Available tools: {len(tools)}")
    # st.dataframe(
    #    tool_list,
    #    use_container_width=True,
    #    hide_index=True,
    #    height=300

    # )


print(st.session_state)
# Agent execution
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandlerChem(
            st.container(),
            max_thought_containers=4,
            collapse_completed_thoughts=False,
            output_placeholder=st.session_state,
        )
        # try:
        # TODO Modify this, not taking callbacks
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
        # except:
        #    st.write("Please input a valid OpenAI API key.")
