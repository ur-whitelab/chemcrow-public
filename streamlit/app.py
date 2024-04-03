import os
import openai
import dotenv
dotenv.load_dotenv(".envgpt4", override=True)
#openai.api_type = "azure"
#openai.api_base = "https://oaiopenaiplaygroundfrancedev.openai.azure.com/"
#openai.api_version = "2023-03-15-preview"
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Init with fake key
if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = 'none'

import pandas as pd
import streamlit as st
from IPython.core.display import HTML
from PIL import Image
from langchain.callbacks import wandb_tracing_enabled
from chemcrow.agents import ChemCrow, make_tools
from chemcrow.frontend.streamlit_callback_handler import \
    StreamlitCallbackHandlerChem

from dotenv import load_dotenv

load_dotenv()
ss = st.session_state

icon = Image.open('assets/logo0.png')
st.set_page_config(
    page_title="MDS Chat",
    page_icon = icon
)

# Set width of sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 550px;
        max-width: 1550px;
    }
    """,
    unsafe_allow_html=True,
)

agent = ChemCrow(
    model='gpt-4',
    tools_model='gpt-4',
    temp=0.1,
    openai_api_key=ss.get('api_key'),
    #api_keys={
    #    'rxn4chem':st.secrets['RXN4CHEM_API_KEY']
    #}
).agent_executor

tools = agent.tools

tool_list = pd.Series(
    {f"✅ {t.name}":t.description for t in tools}
).reset_index()
tool_list.columns = ['Tool', 'Description']

def on_api_key_change():
    api_key = ss.get('api_key') or os.getenv('OPENAI_API_KEY')
    # Check if key is valid
    #if not oai_key_isvalid(api_key):
    #    st.write("Please input a valid OpenAI API key.")

# sidebar
with st.sidebar:
    chemcrow_logo = Image.open('assets/MDSChat.jpg')
    st.image(chemcrow_logo)

    # Input OpenAI api key
    st.markdown('Input your OpenAI API key.')
    st.text_input(
        'OpenAI API key',
        type='password',
        key='api_key',
        on_change=on_api_key_change,
        label_visibility="collapsed"
    )

    # Display available tools
    st.markdown(f"# Available tools: {len(tool_list)}")
    st.dataframe(
        tool_list,
        use_container_width=True,
        hide_index=True,
        height=1500,
        width=1500
    )


    # Initialize or clear conversation history
if 'conversation_history' not in ss:
    ss['conversation_history'] = []
elif st.button('New Chat'):
    ss['conversation_history'] = []

# Agent execution
if prompt := st.chat_input():
    # Append user's message to conversation history
    ss['conversation_history'].append(f"User: {prompt}")
    for message in ss['conversation_history']:
        if message.startswith("User:"):
            st.chat_message("user").write(message[5:])
        else:
            with st.chat_message("assistant"):
                st.write(message[11:])
    st_callback = StreamlitCallbackHandlerChem(
        st.container(),
        max_thought_containers = 4,
        collapse_completed_thoughts = False,
        output_placeholder=st.session_state
    )
    # Concatenate conversation history into a single string
    full_prompt = "\n".join(ss['conversation_history'])
    #with wandb_tracing_enabled():
    response = agent.run(full_prompt, callbacks=[st_callback])
    # Append agent's response to conversation history
    ss['conversation_history'].append(f"Assistant: {response}")

