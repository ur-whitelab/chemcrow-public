import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from chemcrow.agents import ChemCrow
from chemcrow.frontend.streamlit_callback_handler import StreamlitCallbackHandlerChem

load_dotenv()
ss = st.session_state


agent = ChemCrow(
    # tools,
    model="gpt-4",
    temp=0.1,
).agent_executor


tools = agent.tools
tool_list = pd.Series({f"✅ {t.name}": t.description for t in tools}).reset_index()
tool_list.columns = ["Tool", "Description"]


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


def oai_api_key_change():
    api_key = ss.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key


def cs_api_key_change():
    api_key = ss.get("chemspace_api_key") or os.getenv("CHEMSPACE_API_KEY")
    os.environ["CHEMSPACE_API_KEY"] = api_key


# sidebar
with st.sidebar:
    chemcrow_logo = Image.open("assets/chemcrow-logo-bold-new.png")
    st.image(chemcrow_logo)

    # Input OpenAI api key
    st.markdown("Input your OpenAI API key.")
    st.text_input(
        "OpenAI API key",
        type="password",
        key="openai_api_key",
        on_change=oai_api_key_change,
        label_visibility="collapsed",
    )
    # Input ChemSpace API key
    st.markdown("Input your Chemspace API key (optional-used for molecule price).")
    st.text_input(
        "Chemspace API key",
        type="password",
        key="chemspace_api_key",
        on_change=cs_api_key_change,
        label_visibility="collapsed",
    )

    # Display available tools
    st.markdown(f"# Available tools: {len(tools)}")
    st.dataframe(tool_list, use_container_width=True, hide_index=True, height=300)


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
